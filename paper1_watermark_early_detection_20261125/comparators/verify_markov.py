"""Independent deterministic checks for the no-repeat quarter-turn Markov key."""
import json
import numpy as np
import rotation_markov as study


def main():
    rng=np.random.default_rng(4091127)
    uniforms=rng.random((4096,150,50))
    iid,_=study.quarter_keys(uniforms,False)
    nr,codes=study.quarter_keys(uniforms,True)
    # Exact orthogonality and per-message energy preservation.
    identity=np.broadcast_to(np.eye(2),(4096,150,50,2,2))
    for keys in (iid,nr):
        np.testing.assert_array_equal(keys@np.swapaxes(keys,-1,-2),identity)
    if np.any(codes[:,1:]==codes[:,:-1]):
        raise AssertionError("No-repeat key contains an adjacent repeated state")
    transition=np.bincount((codes[:,:-1].ravel()*4+codes[:,1:].ravel()),minlength=16).reshape(4,4)
    row=transition/transition.sum(axis=1,keepdims=True)
    state=np.bincount(codes.ravel(),minlength=4)/codes.size
    expected=np.full((4,4),1/3);np.fill_diagonal(expected,0.)
    assert np.max(np.abs(row-expected))<.004
    assert np.max(np.abs(state-.25))<.004
    vector=rng.normal(size=(4096,150,50,2))
    encoded=np.einsum("btjmn,btjn->btjm",np.swapaxes(iid,-1,-2),vector)
    encoded_norm=np.linalg.norm(encoded,axis=-1)
    original_norm=np.linalg.norm(vector,axis=-1)
    np.testing.assert_allclose(encoded_norm,original_norm,rtol=1e-13,atol=1e-13)
    output={"uniform_marginal_max_abs_error":float(np.max(np.abs(state-.25))),
            "transition_matrix_empirical":row.tolist(),"transition_matrix_theory":expected.tolist(),
            "conditional_entropy_bits":float(np.log2(3)),
            "minimum_transition_probability":float(row[row>0].min()),
            "adjacent_repeat_count":0,"orthogonality_max_abs_error":0.,
            "packet_energy_max_abs_error":float(np.max(np.abs(encoded_norm-original_norm)))}
    study.exp.old.base._atomic_json(study.HERE/"MARKOV_DESIGN_AUDIT.json",output)
    print(json.dumps(output,indent=2))


if __name__=="__main__":main()
