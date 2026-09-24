"""Chained Track B surface layer for ocean/lake and sea ice: PBL `advanc` -> tile fluxes (real-Fortran inputs
only; no recorded PBL outputs are used). Used to validate composition against the real SURFACE tile records."""
import os, sys, numpy as np, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbl_ff as P, pbl_compare as PC, surface_tile_ff as S, ice_props_ff as I


def run_chain(pbl_rec, tile_rec):
    """pbl_rec: (N,154) PBL calls for itype<=2 in call order; tile_rec: (N,90) matching tile records."""
    assert np.array_equal(pbl_rec[:, [0, 1, 2]], tile_rec[:, [0, 1, 2]]), "PBL/tile record order mismatch"
    out = PC.run(pbl_rec)
    d = S.to_inputs(tile_rec)
    # ice properties from tile state (instead of the recorded ones)
    ice = tile_rec[:, 2] == 2
    df, h1, h2, f1, f2 = I.ice_tile_props(d["tg1"], d["tg2"], d["snow"], d["ssi1"], d["ssi2"], d["srheat"],
                                          d["flag_dsws"] > 0.5)
    d.update(dF1dTG=jnp.where(ice, df, 0.0), hcg1=jnp.where(ice, h1, 1.0), hcg2=jnp.where(ice, h2, 1.0),
             fsri1=jnp.where(ice, f1, 0.0), fsri2=jnp.where(ice, f2, 0.0))
    # PBL outputs (ours) replace the recorded ones
    d.update(us=jnp.asarray(out["us"]), vs=jnp.asarray(out["vs"]), ws=jnp.asarray(out["ws"]),
             gusti=jnp.asarray(pbl_rec[:, 113]), qsrf=jnp.asarray(out["qsrf"]), cm=jnp.asarray(out["cm"]),
             ch=jnp.asarray(out["ch"]), cq=jnp.asarray(out["cq"]), ts=jnp.asarray(out["tsv"]),
             dskin=jnp.asarray(out["dskin"]), tsv=jnp.asarray(out["tsv"]))
    # tprime/qprime as computed inside advanc (ddml): recompute from PBL inputs
    ddml = pbl_rec[:, 23] > 0.5
    d["tprime"] = jnp.asarray(np.where(ddml, pbl_rec[:, 25] - pbl_rec[:, 7], 0.0))
    d["qprime"] = jnp.asarray(np.where(ddml, pbl_rec[:, 26] - pbl_rec[:, 39], 0.0))
    return S.tile_fluxes(d)
