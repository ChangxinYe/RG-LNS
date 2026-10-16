"""Shared solver options; importing this module never starts an evaluation."""

from __future__ import annotations

import argparse
from initial_solvers.linear_programming import LPSolverConfig
from evaluation.common.jpwleg_inputs import parse_bool
from initial_solvers.pomeranz import PomeranzSolverConfig


DEFAULT_LP_TOP_K = 1
DEFAULT_LP_REFINEMENT_ITERATIONS = 5
DEFAULT_LP_PROBABILITY_LAMBDA = 5.0
DEFAULT_LP_MINIMUM_WEIGHT = 1e-15
DEFAULT_LP_MATCH_TOLERANCE = 1e-4
DEFAULT_LP_RIGID_COMPONENTS = True
DEFAULT_LP_BUDDY_CHECK = True
DEFAULT_LP_LOOP_CHECK = True
DEFAULT_LP_GREEDY_STOP_THRESHOLD = 1.25
DEFAULT_LP_TIME_LIMIT = 0.0


def add_lp_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--lp-top-k", default=DEFAULT_LP_TOP_K, type=int, help="candidate count kk retained per piece and direction")
    parser.add_argument("--lp-refinement-iterations", default=DEFAULT_LP_REFINEMENT_ITERATIONS, type=int, help="maximum rigid-component optimization rounds after the initial LP")
    parser.add_argument("--lp-probability-lambda", default=DEFAULT_LP_PROBABILITY_LAMBDA, type=float, help="lambda in the probability weight exp(-lambda * ratio^2)")
    parser.add_argument("--lp-minimum-weight", default=DEFAULT_LP_MINIMUM_WEIGHT, type=float, help="numeric cutoff t in the author's code; candidates at the cutoff are removed")
    parser.add_argument("--lp-match-tolerance", default=DEFAULT_LP_MATCH_TOLERANCE, type=float, help="absolute tolerance for LP adjacency equalities")
    parser.add_argument("--lp-rigid-components", default=DEFAULT_LP_RIGID_COMPONENTS, type=parse_bool, help="whether hard equality constraints preserve prior connected-component shapes")
    parser.add_argument("--lp-buddy-check", default=DEFAULT_LP_BUDDY_CHECK, type=parse_bool, help="whether to retain only candidates with reverse-mirror records")
    parser.add_argument("--lp-loop-check", default=DEFAULT_LP_LOOP_CHECK, type=parse_bool, help="whether to run the author's eight four-step cycle checks")
    parser.add_argument("--lp-greedy-stop-threshold", default=DEFAULT_LP_GREEDY_STOP_THRESHOLD, type=float, help="normalized stop threshold for component-level Gallagher merging")
    parser.add_argument("--lp-time-limit", default=DEFAULT_LP_TIME_LIMIT, type=float, help="time limit in seconds for each HiGHS solve; 0 means unlimited")
    return parser


def lp_config_from_args(args: argparse.Namespace) -> LPSolverConfig:
    return LPSolverConfig(
        top_k=args.lp_top_k,
        refinement_iterations=args.lp_refinement_iterations,
        probability_lambda=args.lp_probability_lambda,
        minimum_weight=args.lp_minimum_weight,
        match_tolerance=args.lp_match_tolerance,
        rigid_components=args.lp_rigid_components,
        buddy_check=args.lp_buddy_check,
        loop_check=args.lp_loop_check,
        greedy_stop_threshold=args.lp_greedy_stop_threshold,
        highs_time_limit=args.lp_time_limit,
    )


DEFAULT_POMERANZ_RESTARTS = 10
DEFAULT_POMERANZ_MAX_REFINEMENT_ROUNDS = 0
DEFAULT_POMERANZ_RANDOM_SEED = 0


def add_pomeranz_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--pomeranz-restarts",
        default=DEFAULT_POMERANZ_RESTARTS,
        type=int,
        help="number of distinct initial seeds for independent Pomeranz runs; select the result by final BBM",
    )
    parser.add_argument(
        "--pomeranz-max-refinement-rounds",
        default=DEFAULT_POMERANZ_MAX_REFINEMENT_ROUNDS,
        type=int,
        help="maximum reassembly rounds per seed; 0 runs until BBM no longer strictly improves",
    )
    parser.add_argument(
        "--pomeranz-random-seed",
        default=DEFAULT_POMERANZ_RANDOM_SEED,
        type=int,
        help="fixed random seed for generating initial pieces across restarts",
    )
    return parser


def pomeranz_config_from_args(args: argparse.Namespace) -> PomeranzSolverConfig:
    return PomeranzSolverConfig(
        restarts=args.pomeranz_restarts,
        max_refinement_rounds=args.pomeranz_max_refinement_rounds,
        random_seed=args.pomeranz_random_seed,
    )
