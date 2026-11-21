"""Command-line entry point:  python -m smartattend <command> [options]

  enroll       build the class roster from data/roster/<student_id>/*.jpg
  attend       take attendance from an image, folder, video file or webcam (0)
  analyze      full product on a session: attendance, people count, seating, engagement, annotated video
  stage        reproducible experiment stages (data, engagement, detection, seating, sequence, face, ...)
  math         Requirement 3: tensor / SVD / gradient-check demo
  neurons      Requirement 4: McCulloch-Pitts, sigmoid neuron, MLP + backprop, NumPy vs PyTorch vs TensorFlow
  sandbox      render the tensor / neuron / classic-learning / RBM figures
  experiments  run optimizer comparison, ablations and overfitting study
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from smartattend.utils import get_logger, load_config, set_seed


def _engine(cfg, logger):
    from smartattend.attendance.face import FaceEngine, ensure_models

    det, rec = ensure_models(cfg["paths"]["models_dir"], logger)
    return FaceEngine(det, rec, cfg["face"]["detector_score_threshold"], cfg["face"]["nms_threshold"])


def cmd_enroll(args, cfg, logger) -> int:
    from smartattend.attendance.roster import Roster

    roster_dir = Path(args.roster_dir or cfg["paths"]["roster_dir"])
    if not roster_dir.exists() or not any(p.is_dir() for p in roster_dir.iterdir()):
        logger.error(f"no student folders in {roster_dir}. Expected {roster_dir}/<student_id>/<photo>.jpg")
        return 1
    roster = Roster.enroll_from_dir(roster_dir, _engine(cfg, logger), logger)
    out = Path(args.out or cfg["paths"]["roster_file"])
    roster.save(out)
    logger.info(f"roster with {len(roster)} student(s) saved to {out}")
    return 0 if len(roster) else 1


def cmd_attend(args, cfg, logger) -> int:
    import cv2

    from smartattend.attendance.face import draw_faces
    from smartattend.attendance.roster import Roster
    from smartattend.attendance.session import AttendanceSession, iter_frames

    roster = Roster.load(args.roster or cfg["paths"]["roster_file"])
    engine = _engine(cfg, logger)
    threshold = args.threshold or cfg["face"]["match_threshold"]
    session = AttendanceSession(roster, args.min_frames or cfg["attendance"]["min_frames_present"],
                                cfg["attendance"].get("min_fraction_present", 0.5))
    every = args.every or cfg["attendance"]["sample_every_n_frames"]
    annot_dir = Path(args.save_annotated) if args.save_annotated else None
    if annot_dir:
        annot_dir.mkdir(parents=True, exist_ok=True)

    latencies = []
    for idx, frame in iter_frames(args.input, every):
        t0 = time.perf_counter()
        faces = engine.analyze(frame)
        labels, scores = roster.match([f.embedding for f in faces], threshold) if faces else ([], [])
        latencies.append((time.perf_counter() - t0) * 1000)
        session.update(idx, labels, scores)
        logger.info(f"frame {idx}: {len(faces)} face(s), recognised {[l for l in labels if l]}")
        if annot_dir:
            cv2.imwrite(str(annot_dir / f"frame_{idx:06d}.jpg"), draw_faces(frame, faces, labels, scores))

    out = session.to_csv(args.out)
    summary = session.summary()
    if latencies:
        summary["latency_ms_mean"] = round(sum(latencies) / len(latencies), 1)
        summary["latency_ms_max"] = round(max(latencies), 1)
    logger.info("summary " + json.dumps(summary))
    logger.info(f"attendance register written to {out}")
    return 0


def cmd_sandbox(args, cfg, logger) -> int:
    from smartattend.nnfund import boltzmann, classic_learning, neurons, tensor_sandbox

    base = Path(cfg["paths"]["figures_dir"]) / "nnfund"
    for name, mod in [("tensor_sandbox", tensor_sandbox), ("neurons", neurons),
                      ("classic_learning", classic_learning), ("boltzmann", boltzmann)]:
        for p in mod.make_figures(base):
            logger.info(f"[{name}] wrote {p}")
    return 0


def cmd_math(args, cfg, logger) -> int:
    """Requirement 3: print the worked tensor examples and draw the sandbox figures."""
    from smartattend.nnfund import tensor_sandbox

    print("\n".join(tensor_sandbox.demo()))
    for p in tensor_sandbox.make_figures(Path(cfg["paths"]["figures_dir"]) / "nnfund"):
        print(f"figure: {p}")
    return 0


def cmd_neurons(args, cfg, logger) -> int:
    """Requirement 4: McCulloch-Pitts, sigmoid neuron, MLP + backprop, NumPy vs PyTorch vs TensorFlow."""
    from smartattend.nnfund.neural_networks import run

    paths = cfg["paths"]
    lines, _ = run(Path(paths["data_processed"]) / "kaggle_classroom", paths["eval_dir"],
                   Path(paths["figures_dir"]) / "nnfund", epochs=args.epochs, with_keras=not args.no_keras)
    print("\n".join(lines))
    return 0


def cmd_optimize(args, cfg, logger) -> int:
    """Requirement 5: optimizers and ablations on real face frames."""
    from smartattend.nnfund.optimization import run

    paths = cfg["paths"]
    lines, _ = run(Path(paths["data_raw"]) / "drowsiness", paths["eval_dir"], Path(paths["figures_dir"]) / "nnfund",
                   epochs=args.epochs, seed=cfg["seed"], logger=logger)
    print("\n".join(lines))
    return 0


def cmd_classic(args, cfg, logger) -> int:
    """Requirement 6: Hebbian / Oja / Sanger, competitive, error-correction learning and RBM on real data."""
    from smartattend.nnfund.classic_real import run
    from smartattend.nnfund.optimization import load_faces

    paths = cfg["paths"]
    load_faces(Path(paths["data_raw"]) / "drowsiness", seed=cfg["seed"])          # builds the 32x32 cache if missing
    opt_json = Path(paths["eval_dir"]) / "optimization.json"
    mlp_ref = None
    if opt_json.exists():
        mlp_ref = json.loads(opt_json.read_text())["ablations"]["Regularisation"]["none (baseline)"]["test_f1_last"][0]
    lines, _ = run(Path(paths["data_processed"]) / "drowsiness_faces32.npz",
                   Path(paths["data_processed"]) / "lfw" / "emb_clean.npz", paths["eval_dir"],
                   Path(paths["figures_dir"]) / "nnfund", mlp_reference=mlp_ref)
    print("\n".join(lines))
    return 0


def cmd_classroom(args, cfg, logger) -> int:
    """Requirement 7: detection, per-student states, trends and convolution figures on the Kaggle test classroom."""
    from smartattend.vision.classroom_eval import run

    lines, _ = run(cfg, epochs=args.epochs, log=logger.info)
    print("\n".join(lines))
    return 0


def cmd_benchmark(args, cfg, logger) -> int:
    """Requirement 8: R-CNN, Fast R-CNN, Faster R-CNN, RetinaNet, YOLO, ResNet, GoogLeNet on the Kaggle test classroom."""
    from smartattend.vision.benchmark8 import run

    lines, _ = run(cfg, log=logger.info, fast_epochs=args.fast_epochs, tv_epochs=args.tv_epochs)
    print("\n".join(lines))
    return 0


def cmd_sequence(args, cfg, logger) -> int:
    """Requirement 9: LSTM vs Transformer on real classroom video, attention, latent spaces, step/sigmoid/tanh."""
    from smartattend.sequence.classroom_seq import run

    lines, _ = run(cfg, log=logger.info, with_keras=not args.no_keras)
    print("\n".join(lines))
    return 0


def cmd_evaluate(args, cfg, logger) -> int:
    """Requirement 10: robustness, fairness, error analysis, latency and the scorecard on the Kaggle test classroom."""
    from smartattend.eval.final_eval import run

    lines, _ = run(cfg, log=logger.info)
    print("\n".join(lines))
    return 0


def cmd_experiments(args, cfg, logger) -> int:
    from smartattend.nnfund.experiments import run_all

    run_all({**cfg["nnfund"], "seed": cfg["seed"]}, Path(cfg["paths"]["experiments_dir"]) / "nnfund", logger)
    return 0


def cmd_stage(args, cfg, logger) -> int:
    from smartattend.stages import run_stage

    run_stage(args.name, cfg, logger)
    return 0


def cmd_analyze(args, cfg, logger) -> int:
    from smartattend.attendance.roster import Roster
    from smartattend.pipeline import process_source

    roster_path = Path(args.roster or cfg["paths"]["roster_file"])
    roster = Roster.load(roster_path) if roster_path.exists() else None
    if roster is None:
        logger.warning("no roster found - running without identity recognition (counts and engagement only)")
    if args.detector:
        cfg["product"]["detector"] = args.detector
    summary = process_source(cfg, args.input, args.out, roster, args.every or cfg["attendance"]["sample_every_n_frames"],
                             save_video=not args.no_video, logger=logger)
    logger.info("session summary " + json.dumps(summary))
    return 0


def cmd_acceptance(args, cfg, logger) -> int:
    from smartattend.requirements import evaluate, format_table
    from smartattend.utils.config import PROJECT_ROOT

    rows = evaluate(PROJECT_ROOT, cfg)
    print(format_table(rows))
    out = Path(cfg["paths"]["eval_dir"]) / "acceptance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {out}")
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smartattend", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("enroll", help="build roster embeddings from student photos")
    e.add_argument("--roster-dir")
    e.add_argument("--out")
    e.set_defaults(func=cmd_enroll)

    a = sub.add_parser("attend", help="take attendance from image / folder / video / webcam")
    a.add_argument("--input", required=True, help="image, folder, video file, or webcam index like 0")
    a.add_argument("--roster")
    a.add_argument("--out", default="reports/attendance.csv")
    a.add_argument("--every", type=int, help="process 1 of every N video frames")
    a.add_argument("--min-frames", type=int)
    a.add_argument("--threshold", type=float, help="cosine similarity threshold")
    a.add_argument("--save-annotated", help="folder for annotated frames")
    a.set_defaults(func=cmd_attend)

    z = sub.add_parser("analyze", help="full product: attendance + seating + engagement on a session")
    z.add_argument("--input", required=True, help="image, folder, video file, or webcam index like 0")
    z.add_argument("--roster")
    z.add_argument("--out", default="reports/session")
    z.add_argument("--every", type=int)
    z.add_argument("--detector", help="override product detector, e.g. yolov8s or fasterrcnn_resnet50_fpn")
    z.add_argument("--no-video", action="store_true")
    z.set_defaults(func=cmd_analyze)

    from smartattend.stages import STAGES

    s = sub.add_parser("stage", help="run a reproducible experiment stage")
    s.add_argument("name", choices=[*STAGES, "all"])
    s.set_defaults(func=cmd_stage)

    sub.add_parser("acceptance", help="check every requirement target (Requirement 1)").set_defaults(func=cmd_acceptance)
    sub.add_parser("math", help="Requirement 3: tensor / SVD / gradient-check demo + figures").set_defaults(func=cmd_math)
    n = sub.add_parser("neurons", help="Requirement 4: neurons, MLP + backprop, framework check")
    n.add_argument("--epochs", type=int, default=40)
    n.add_argument("--no-keras", action="store_true", help="skip the TensorFlow/Keras comparison")
    n.set_defaults(func=cmd_neurons)
    o = sub.add_parser("optimize", help="Requirement 5: optimizers + init/normalisation/regularisation/augmentation ablations")
    o.add_argument("--epochs", type=int, default=30)
    o.set_defaults(func=cmd_optimize)
    sub.add_parser("classic", help="Requirement 6: Hebbian/Oja/Sanger, competitive, error-correction, RBM on real data"
                   ).set_defaults(func=cmd_classic)
    c = sub.add_parser("classroom", help="Requirement 7: evaluate detection, states and trends on the Kaggle test classroom")
    c.add_argument("--epochs", type=int, default=6, help="fine-tuning epochs for the ResNet18 / GoogLeNet state models")
    c.set_defaults(func=cmd_classroom)
    b = sub.add_parser("benchmark", help="Requirement 8: benchmark all detector families + ResNet/GoogLeNet")
    b.add_argument("--fast-epochs", type=int, default=6, help="Fast R-CNN head training epochs")
    b.add_argument("--tv-epochs", type=int, default=3, help="Faster R-CNN / RetinaNet fine-tuning epochs")
    b.set_defaults(func=cmd_benchmark)
    q = sub.add_parser("sequence", help="Requirement 9: LSTM / Transformer / attention / latent space on classroom video")
    q.add_argument("--no-keras", action="store_true", help="skip the TensorFlow/Keras LSTM")
    q.set_defaults(func=cmd_sequence)
    sub.add_parser("evaluate", help="Requirement 10: final evaluation (robustness, fairness, errors, latency, scorecard)"
                   ).set_defaults(func=cmd_evaluate)
    sub.add_parser("sandbox", help="render fundamentals figures").set_defaults(func=cmd_sandbox)
    sub.add_parser("experiments", help="optimizers, ablations, overfitting").set_defaults(func=cmd_experiments)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger("smartattend", cfg["paths"]["logs_dir"])
    logger.info(f"command={args.command} seed={cfg['seed']}")
    return args.func(args, cfg, logger)
