// Dry-run: drive the DEPLOYED X2PolicyPipeline from a running
// planner_refsrc_x2.py stream and report what the policy would command.
//
// PUBLISHES NOTHING.  No ROS, no HAL, no command topic is linked into this
// binary at all -- the robot cannot move from running it.  The point is to
// prove the on-robot planner stream is accepted by the real pipeline before
// any command-capable runtime is pointed at it.
//
// The robot state is held STATIC at the bundle default pose, so tracking error
// grows on purpose: this checks stream acceptance, observation well-formedness,
// action bounds and timing -- not closed-loop tracking, which the MuJoCo
// harness on the workstation already covers.
//
// Build & run (on the robot, from x2_rl_deploy_controller/):
//   g++ -O2 -std=c++17 -Iinclude -I/usr/local/include/onnxruntime \
//       src/x2_policy_pipeline.cc test/x2_refsrc_dryrun.cc \
//       -L/usr/local/lib -lonnxruntime -lpthread -o /tmp/x2_refsrc_dryrun
//   /tmp/x2_refsrc_dryrun config/rl_model/x2_bundle \
//       config/rl_model/x2_policy_x2loco_s4000.onnx 127.0.0.1 5602 500
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "x2_rl_deploy_controller/x2_policy_pipeline.h"

namespace {

std::vector<float> ReadF32(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(2);
  }
  f.seekg(0, std::ios::end);
  std::vector<float> v(static_cast<size_t>(f.tellg()) / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size() * 4));
  return v;
}

struct Stat {
  double lo = 1e30, hi = -1e30, absmax = 0;
  bool finite = true;
  void add(float x) {
    if (!std::isfinite(x)) { finite = false; return; }
    lo = std::min<double>(lo, x);
    hi = std::max<double>(hi, x);
    absmax = std::max<double>(absmax, std::fabs(x));
  }
};

double Pct(std::vector<double> &v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))];
}

}  // namespace

int main(int argc, char **argv) {
  const std::string bundle = argc > 1 ? argv[1] : "config/rl_model/x2_bundle";
  const std::string model  = argc > 2 ? argv[2] : "config/rl_model/x2_policy_x2loco_s4000.onnx";
  const std::string host   = argc > 3 ? argv[3] : "127.0.0.1";
  const int port           = argc > 4 ? std::atoi(argv[4]) : 5602;
  const int steps          = argc > 5 ? std::atoi(argv[5]) : 500;

  x2::X2PolicyPipeline pipe;
  if (!pipe.Load(bundle)) {
    std::fprintf(stderr, "bundle load failed: %s\n", bundle.c_str());
    return 2;
  }
  const auto def31 = ReadF32(bundle + "/default.f32");
  const auto lim_lo = ReadF32(bundle + "/lim_lo.f32");
  const auto lim_hi = ReadF32(bundle + "/lim_hi.f32");
  std::printf("bundle ok: default=%zu lim=%zu/%zu\n", def31.size(), lim_lo.size(), lim_hi.size());

  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "dryrun");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(2);
  Ort::Session sess(env, model.c_str(), so);
  Ort::AllocatorWithDefaultOptions alloc;
  const auto in_name = sess.GetInputNameAllocated(0, alloc);
  const auto out_name = sess.GetOutputNameAllocated(0, alloc);
  const auto in_shape = sess.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const auto out_shape = sess.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  std::printf("policy: %s%s -> %s%s\n", in_name.get(),
              in_shape.size() == 2 ? ("[1," + std::to_string(in_shape[1]) + "]").c_str() : "",
              out_name.get(),
              out_shape.size() == 2 ? ("[1," + std::to_string(out_shape[1]) + "]").c_str() : "");

  if (!pipe.EnableLive(host, port)) {
    std::fprintf(stderr, "EnableLive failed\n");
    return 2;
  }
  std::printf("connecting to %s:%d ...\n", host.c_str(), port);

  // Static robot state: standing at the bundle default pose, upright, still.
  const std::vector<float> q_il(def31.begin(), def31.end());
  const std::vector<float> v_il(q_il.size(), 0.f);
  const float quat[4] = {1.f, 0.f, 0.f, 0.f};
  const float ang[3] = {0.f, 0.f, 0.f};

  // AlignHeading needs a real frame; the future window needs 25 more.
  int waited = 0;
  while (!pipe.AlignHeading(quat)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (++waited > 500) {
      std::fprintf(stderr, "no reference frame after 10 s -- is planner_refsrc_x2.py running?\n");
      return 3;
    }
  }
  std::printf("heading aligned after %d ms of stream\n", waited * 20);

  const char *in_names[] = {in_name.get()};
  const char *out_names[] = {out_name.get()};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Stat s_obs, s_act, s_tgt;
  std::vector<double> t_obs, t_inf, t_stale;
  int clamped = 0, out_of_limit = 0, n = 0;
  std::vector<float> pos31, kp31, kd31;

  auto next = std::chrono::steady_clock::now();
  for (int i = 0; i < steps; ++i) {
    next += std::chrono::milliseconds(20);          // one policy step = 50 Hz

    auto a0 = std::chrono::steady_clock::now();
    std::vector<float> obs = pipe.BuildObs(q_il, v_il, quat, ang);
    auto a1 = std::chrono::steady_clock::now();
    if (obs.empty()) {                              // waiting for the window
      std::this_thread::sleep_until(next);
      continue;
    }
    for (float x : obs) s_obs.add(x);

    std::array<int64_t, 2> shape{1, static_cast<int64_t>(obs.size())};
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(),
                                                    shape.data(), shape.size());
    auto outs = sess.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    auto a2 = std::chrono::steady_clock::now();

    const float *raw = outs[0].GetTensorData<float>();
    const size_t na = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> action(na);
    for (size_t j = 0; j < na; ++j) {
      action[j] = x2::X2PolicyPipeline::ClampAction(raw[j]);
      if (action[j] != raw[j]) ++clamped;
      s_act.add(raw[j]);
    }

    pipe.Targets(action, q_il, pos31, kp31, kd31);
    for (size_t j = 0; j < pos31.size(); ++j) {
      s_tgt.add(pos31[j]);
      if (j < lim_lo.size() && (pos31[j] < lim_lo[j] - 1e-4f || pos31[j] > lim_hi[j] + 1e-4f))
        ++out_of_limit;
    }

    t_obs.push_back(std::chrono::duration<double, std::milli>(a1 - a0).count());
    t_inf.push_back(std::chrono::duration<double, std::milli>(a2 - a1).count());
    t_stale.push_back(pipe.LiveStaleSecs());
    ++n;
    std::this_thread::sleep_until(next);
  }
  pipe.StopLive();

  std::printf("\n--- %d policy steps -------------------------------------------\n", n);
  std::printf("obs    finite=%s  range %+.3f .. %+.3f  |max|=%.3f  (dim %zu)\n",
              s_obs.finite ? "yes" : "NO", s_obs.lo, s_obs.hi, s_obs.absmax,
              static_cast<size_t>(in_shape.size() == 2 ? in_shape[1] : 0));
  std::printf("action finite=%s  range %+.3f .. %+.3f  |max|=%.3f  clamped=%d\n",
              s_act.finite ? "yes" : "NO", s_act.lo, s_act.hi, s_act.absmax, clamped);
  std::printf("target finite=%s  range %+.3f .. %+.3f  outside hard limits=%d\n",
              s_tgt.finite ? "yes" : "NO", s_tgt.lo, s_tgt.hi, s_tgt.absmax > 0 ? out_of_limit : 0);
  {
    auto tmp = t_inf;
    const int over = static_cast<int>(std::count_if(tmp.begin(), tmp.end(),
                                                    [](double x) { return x > 20.0; }));
    const double worst = *std::max_element(tmp.begin(), tmp.end());
    const double first = t_inf.empty() ? 0 : t_inf.front();
    std::printf("timing BuildObs p50=%.2f p99=%.2f ms | inference p50=%.2f p90=%.2f "
                "p99=%.2f max=%.2f ms  over-20ms=%d/%d  first-call=%.2f\n",
                Pct(t_obs, .5), Pct(t_obs, .99), Pct(t_inf, .5), Pct(t_inf, .9),
                Pct(t_inf, .99), worst, over, n, first);
  }
  std::printf("stream staleness p50=%.3f p99=%.3f s\n", Pct(t_stale, .5), Pct(t_stale, .99));

  const bool ok = s_obs.finite && s_act.finite && s_tgt.finite && out_of_limit == 0 &&
                  Pct(t_inf, .99) + Pct(t_obs, .99) < 20.0 && Pct(t_stale, .99) < 0.2;
  std::printf("\n%s (no command was published)\n", ok ? "DRY RUN OK" : "DRY RUN FLAGGED ISSUES");
  return ok ? 0 : 1;
}
