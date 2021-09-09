"""
Benchmark loader methods
"""

from typing import List, Tuple

from guacamol.goal_directed_benchmark import GoalDirectedBenchmark

from egegl.scoring.benchmarks_custom import (
    penalized_logp_atomrings,
    penalized_logp_cyclebasis,
)
from egegl.scoring.benchmarks_guacamol import (
    amlodipine_rings,
    cns_mpo,
    decoration_hop,
    hard_fexofenadine,
    hard_osimertinib,
    isomers_c7h8n2o2,
    isomers_c9h10n2o2pf2cl,
    isomers_c11h24,
    logP_benchmark,
    median_camphor_menthol,
    median_tadalafil_sildenafil,
    perindopril_rings,
    pioglitazone_mpo,
    qed_benchmark,
    ranolazine_mpo,
    scaffold_hop,
    similarity,
    sitagliptin_replacement,
    tpsa_benchmark,
    valsartan_smarts,
    zaleplon_with_other_formula,
)


def load_benchmark(benchmark_id: int) -> Tuple[GoalDirectedBenchmark, List[int]]:
    benchmark = {
        0: similarity(
            smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
            name="Celecoxib",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        1: similarity(
            smiles="Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
            name="Troglitazone",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        2: similarity(
            smiles="CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
            name="Thiothixene",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
        3: similarity(
            smiles="Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
            name="Aripiprazole",
            fp_type="ECFP4",
            threshold=0.75,
        ),
        4: similarity(
            smiles="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
            name="Albuterol",
            fp_type="FCFP4",
            threshold=0.75,
        ),
        5: similarity(
            smiles="COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
            name="Mestranol",
            fp_type="AP",
            threshold=0.75,
        ),
        6: isomers_c11h24(),
        7: isomers_c9h10n2o2pf2cl(),
        8: median_camphor_menthol(),
        9: median_tadalafil_sildenafil(),
        10: hard_osimertinib(),
        11: hard_fexofenadine(),
        12: ranolazine_mpo(),
        13: perindopril_rings(),
        14: amlodipine_rings(),
        15: sitagliptin_replacement(),
        16: zaleplon_with_other_formula(),
        17: valsartan_smarts(),
        18: decoration_hop(),
        19: scaffold_hop(),
        20: logP_benchmark(target=-1.0),
        21: logP_benchmark(target=8.0),
        22: tpsa_benchmark(target=150.0),
        23: cns_mpo(),
        24: qed_benchmark(),
        25: isomers_c7h8n2o2(),
        26: pioglitazone_mpo(),
        27: penalized_logp_atomrings(),
        28: penalized_logp_cyclebasis(),
    }.get(benchmark_id)

    if benchmark_id in [
        3,
        4,
        5,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        26,
    ]:
        scoring_num_list = [1, 10, 100]
    elif benchmark_id in [6]:
        scoring_num_list = [159]
    elif benchmark_id in [7]:
        scoring_num_list = [250]
    elif benchmark_id in [25]:
        scoring_num_list = [100]
    elif benchmark_id in [0, 1, 2, 27, 28]:
        scoring_num_list = [1]

    return benchmark, scoring_num_list  # type: ignore

