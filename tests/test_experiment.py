"""End-to-end test of dataset building, caching and evaluation on synthetic files.

The ESC-50 / LibriSpeech layouts are imitated with generated tones and noise,
and the model is a test double. No real audio or model is involved.
"""


from sonic import dataset as ds
from sonic.config import load_config, replace_section
from sonic.dataset_builder import build_split, list_librispeech, load_esc50, split_pools
from sonic.experiment import cache_split, evaluate_config, load_cached
from tests.conftest import EnergyModel

SR = 32000


def test_build_cache_and_evaluate(fake_sources, tmp_path):
    esc_root, libri_root = fake_sources
    cfg = load_config(overrides=[f"paths.data_root={tmp_path / 'data'}",
                                 f"paths.cache_dir={tmp_path / 'cache'}"])
    pools = split_pools(load_esc50(esc_root), list_librispeech(libri_root))
    assert set(pools["eval"].esc.fold) == {5}
    assert not set(pools["dev"].speech) & set(pools["eval"].speech)

    manifest = build_split(pools["eval"], "eval", tmp_path / "data", n_per_scenario=2, cfg=cfg.dataset)
    assert len(manifest) == 6 and set(manifest.scenario) == set(ds.SCENARIOS)

    model = EnergyModel()
    cache_split(model, cfg, "eval", window_sizes=[0.5, 1.0], log=lambda *_: None)
    cs = load_cached(cfg, "eval", manifest.sample_id[0])
    assert {"w0.5", "w1"} <= set(cs.scores)
    assert "separation_s" in cs.timings

    df = evaluate_config(cfg, "eval", output_dir=tmp_path / "out")
    assert len(df) == 18
    assert set(df.method) == {"baseline", "sonic", "oracle"}
    assert (tmp_path / "out" / manifest.sample_id[0] / "sonic.wav").exists()
    # baseline masks everything; localized methods must mask strictly less
    assert (df[df.method == "baseline"].masked_fraction == 1).all()
    assert (df[df.method == "oracle"].masked_fraction < 1).all()

    small = replace_section(cfg, "localization", win_s=0.5)
    df2 = evaluate_config(small, "eval", methods=["sonic"])
    assert len(df2) == 6


def test_ablations_select_best_and_plots(fake_sources, tmp_path):
    from sonic.ablation import ABLATION_WINDOWS, run_ablations, select_best, variants
    from sonic.evaluation import aggregate
    from sonic.plotting import plot_ablation, plot_localization, plot_method_bars

    esc_root, libri_root = fake_sources
    cfg = load_config(overrides=[f"paths.data_root={tmp_path / 'data'}",
                                 f"paths.cache_dir={tmp_path / 'cache'}"])
    pools = split_pools(load_esc50(esc_root), list_librispeech(libri_root))
    build_split(pools["dev"], "dev", tmp_path / "data", n_per_scenario=1, cfg=cfg.dataset)
    cache_split(EnergyModel(), cfg, "dev", window_sizes=ABLATION_WINDOWS, log=lambda *_: None)

    df = run_ablations(cfg, "dev", log=lambda *_: None)
    assert set(df.factor) == {"win_s", "threshold", "crossfade_s", "mask", "reference"}
    assert len(df[df.factor != "reference"]) == len(variants(cfg)) * 3

    tuned, overrides = select_best(df, cfg)
    assert any(o.startswith("localization.threshold=") for o in overrides)
    assert tuned.with_overrides(overrides) == tuned

    figs = tmp_path / "figs"
    plot_ablation(df, "threshold", ["sisdr_bg", "loc_f1"], figs / "abl.png")
    plot_ablation(df, "mask", ["sisdr_bg"], figs / "mask.png")
    ref = df[df.factor == "reference"]
    summary = aggregate(ref, ["sisdr_bg"], by=["scenario", "method"])
    plot_method_bars(summary, ["sisdr_bg"], figs / "bars.png")
    cs = load_cached(cfg, "dev", df.sample_id.iloc[0])
    from sonic.experiment import apply_method
    out = apply_method(cs, "sonic", cfg)
    loc = out.localization
    plot_localization(cs.sample.x, SR, loc.frame_times, loc.frame_scores, loc.threshold,
                      out.intervals, cs.sample.intervals, out.mask, "t", figs / "loc.png")
    assert all((figs / f).exists() for f in ("abl.png", "mask.png", "bars.png", "loc.png"))
