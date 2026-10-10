"""Offline contract tests for the solver-informed agent (no model API needed)."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solver_informed_travel_agent as agent  # noqa: E402


class TypedIRTests(unittest.TestCase):
    def base_ir(self):
        return {
            "trip": {"origin": "杭州", "destination": "绍兴", "start_date": "2025-11-12",
                     "end_date": "2025-11-13", "travelers": 1, "rooms": 1},
            "transport": {"outbound": {"allowed_modes": ["train"], "constraints": [
                {"field": "departure_time", "op": ">=", "value": "06:00", "source_text": "6点后"}]},
                "return": {"allowed_modes": ["train"], "constraints": []}},
            "hotel": {"constraints": [
                {"field": "decoration_year", "op": ">", "value": 2025, "source_text": "2025年之后"}],
                "soft_preferences": [
                {"field": "services", "op": "contains", "value": "电视投屏", "source_text": "最好有电视投屏"},
                ],
                "objective": {"field": "rating", "sense": "max", "source_text": "评分最高"}},
            "attractions": {"selection": "all_matching", "constraints": [
                {"field": "price", "op": "==", "value": 0, "source_text": "所有免费景点"},
                {"field": "attraction_type", "op": "==", "value": "自然风光", "source_text": "所有自然风光景点"}]},
            "restaurants": {"requirements": []},
        }

    def test_normalizes_types_and_semantics(self):
        ir = self.base_ir()
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        self.assertEqual(ir["transport"]["outbound"]["constraints"][0]["value"], 360)
        self.assertEqual(ir["hotel"]["constraints"][0]["op"], ">")
        self.assertEqual(len(ir["hotel"]["soft_preferences"]), 1)
        self.assertEqual(ir["hotel"]["objective"]["sense"], "max")
        self.assertEqual(len(ir["attractions"]["selection_groups"]), 2)

    def test_numeric_time_comparison_is_not_string_comparison(self):
        ir = self.base_ir()
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        boundary = ir["transport"]["outbound"]["constraints"][0]
        self.assertTrue(agent._compare(400, boundary["op"], boundary["value"]))
        self.assertFalse(agent._compare(300, boundary["op"], boundary["value"]))

    def test_normalizes_category_without_reinterpreting_anchor(self):
        ir = self.base_ir()
        ir["attractions"]["constraints"][1]["value"] = "自然风光类"
        ir["restaurants"]["requirements"] = [{
            "required_name": None, "nearby_location": "行程附近", "constraints": []
        }]
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        group_values = [group[0]["value"] for group in ir["attractions"]["selection_groups"]]
        self.assertIn("自然风光", group_values)
        self.assertEqual(ir["restaurants"]["requirements"][0]["nearby_location"], "行程附近")

    def test_validator_does_not_infer_superlative_groups_from_query(self):
        ir = self.base_ir()
        ir["attractions"] = {"selection": "default", "constraints": []}
        query = "安排评分最高的‘自然风光’类景点，再选景点推荐里评分最高的那三个地方"
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        self.assertEqual(ir["attractions"]["selection"], "default")
        self.assertIsNone(ir["attractions"]["count"])
        self.assertNotIn("required_groups", ir["attractions"])

    def test_accepts_midnight_as_interval_endpoint(self):
        self.assertEqual(agent._parse_hhmm("24:00"), 1440)
        self.assertEqual(agent._parse_hhmm(1440), 1440)

    def test_validator_does_not_repair_omitted_comparative_objectives(self):
        ir = self.base_ir()
        ir["restaurants"]["requirements"] = [{
            "required_name": None, "nearby_location": "八一广场", "constraints": []
        }]
        query = "去程坐火车。回程最好直达，行程时间要尽量短。八一广场附近选有等位区的，评分高优先。"
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        self.assertIsNone(ir["transport"]["return"]["objective"])
        self.assertIsNone(ir["restaurants"]["requirements"][0]["objective"])

    def test_accepts_source_grounded_semantic_revision(self):
        original = self.base_ir()
        query = "去程选择最早出发的火车，酒店必须是2025年之后装修。"
        agent.SolverInformedTravelAgent._validate_problem_ir(original)
        candidate = agent.copy.deepcopy(original)
        candidate["transport"]["outbound"]["objective"] = {
            "field": "departure_time", "sense": "min", "source_text": "最早出发",
        }
        revised = agent.SolverInformedTravelAgent._validate_semantic_revision(
            original, candidate, query,
        )
        self.assertEqual(revised["transport"]["outbound"]["objective"]["field"],
                         "departure_time")

    def test_rejects_ungrounded_or_core_semantic_revision(self):
        original = self.base_ir()
        query = "去程坐火车，评分最高的酒店。"
        agent.SolverInformedTravelAgent._validate_problem_ir(original)
        invented = agent.copy.deepcopy(original)
        invented["hotel"]["objective"] = {
            "field": "price", "sense": "min", "source_text": "最便宜的酒店",
        }
        with self.assertRaises(ValueError):
            agent.SolverInformedTravelAgent._validate_semantic_revision(original, invented, query)
        changed_trip = agent.copy.deepcopy(original)
        changed_trip["trip"]["travelers"] = 9
        with self.assertRaises(ValueError):
            agent.SolverInformedTravelAgent._validate_semantic_revision(original, changed_trip, query)

    def test_rejects_unsupported_constraint_field(self):
        ir = self.base_ir()
        ir["attractions"]["constraints"].append({
            "field": "source", "op": "==", "value": "recommendations",
            "source_text": "推荐里的",
        })
        with self.assertRaises(ValueError):
            agent.SolverInformedTravelAgent._validate_problem_ir(ir)

    def test_preserves_typed_objective_and_scopes_named_plus_top_k(self):
        ir = self.base_ir()
        ir["hotel"]["constraints"] = [
            constraint for constraint in ir["hotel"]["constraints"]
            if constraint["field"] != "rating"
        ]
        ir["hotel"]["objective"] = {
            "field": "rating", "sense": "max", "source_text": "性价比高",
        }
        ir["attractions"] = {
            "selection": "named_and_top_k", "count": None,
            "required_names": ["西岸美术馆"],
            "constraints": [{
                "field": "attraction_type", "op": "==", "value": "城市地标",
                "source_text": "城市地标类里评分最高的景点",
            }],
            "objective": {
                "field": "rating", "sense": "max", "top_k": 1,
                "source_text": "评分最高的景点",
            },
        }
        query = "酒店性价比高，西岸美术馆一定要去，再去城市地标类里评分最高的景点。"
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        self.assertEqual(ir["hotel"]["objective"]["field"], "rating")
        self.assertEqual(ir["attractions"]["selection"], "named")
        self.assertEqual(ir["attractions"]["constraints"], [])
        self.assertEqual(ir["attractions"]["required_groups"][0]["count"], 1)

    def test_scopes_named_plus_filtered_objective_without_query_regex(self):
        ir = self.base_ir()
        ir["attractions"] = {
            "selection": "named", "count": None,
            "required_names": ["西岸美术馆", "正大广场"],
            "constraints": [{
                "field": "attraction_type", "op": "==", "value": "城市地标",
                "source_text": "城市地标类里评分最高的景点",
            }],
            "objective": {
                "field": "rating", "sense": "max", "source_text": "评分最高的景点",
            },
        }
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        self.assertEqual(ir["attractions"]["selection"], "named")
        self.assertEqual(ir["attractions"]["constraints"], [])
        self.assertEqual(ir["attractions"]["required_groups"][0]["count"], 1)

    def test_rejects_removal_of_explicit_mandatory_source(self):
        original = self.base_ir()
        original["hotel"]["constraints"] = [{
            "field": "services", "op": "contains", "value": "泳池",
            "source_text": "必须有泳池",
        }]
        query = "酒店必须有泳池，去程坐火车。"
        agent.SolverInformedTravelAgent._validate_problem_ir(original)
        candidate = agent.copy.deepcopy(original)
        candidate["hotel"]["constraints"] = []
        with self.assertRaises(ValueError):
            agent.SolverInformedTravelAgent._validate_semantic_revision(original, candidate, query)


class ConstraintEngineTests(unittest.TestCase):
    def minimal_ir(self):
        return {
            "trip": {"origin": "A", "destination": "B", "start_date": "2025-11-12",
                     "end_date": "2025-11-13", "travelers": 2, "rooms": 1,
                     "budget_max": 500},
            "transport": {
                "outbound": {"allowed_modes": ["train"], "constraints": [],
                             "soft_preferences": [], "objective": {"field": "departure_time", "sense": "min"}},
                "return": {"allowed_modes": ["train"], "constraints": [],
                           "soft_preferences": [], "objective": None},
            },
            "hotel": {"constraints": [], "soft_preferences": [], "objective": None},
            "attractions": {"selection": "default", "count": None, "required_names": [],
                            "constraints": [], "soft_preferences": [], "objective": None},
            "restaurants": {"requirements": []},
        }

    def test_compiles_typed_constraints(self):
        constraints = agent.ConstraintCompiler.compile(self.minimal_ir())
        kinds = {constraint.kind for constraint in constraints}
        self.assertIn("candidate_coverage", kinds)
        self.assertIn("arg_extreme", kinds)
        self.assertIn("budget_lower_bound", kinds)
        self.assertNotIn("implication", kinds)
        self.assertNotIn("slot_capacity", kinds)

    def test_budget_lower_bound_can_prove_false(self):
        class Planner:
            problem_ir = self.minimal_ir()

            @staticmethod
            def _trip_days():
                return 2

            @staticmethod
            def _status_against(record, constraints):
                return agent.SAT

            @staticmethod
            def _rank_records(kind, direction=None):
                if kind == "train":
                    return [agent.EvidenceRecord(direction, "train", "test", {}, {"price": 150})]
                if kind == "hotel":
                    return [agent.EvidenceRecord("hotel", "hotel", "test", {}, {"price": 250})]
                return []

            @staticmethod
            def _initial_requests():
                return [
                    agent.AcquisitionRequest("query_train_info", "test", {}, "outbound:train"),
                    agent.AcquisitionRequest("query_train_info", "test", {}, "return:train"),
                    agent.AcquisitionRequest("query_hotel_info", "test", {}, "hotels:test"),
                ]

        planner = Planner()
        planner.state = agent.TravelState(planner.problem_ir)
        for request in planner._initial_requests():
            planner.state.query_outcomes[request.key] = agent.QueryOutcome(
                agent.QUERY_SUCCESS_NONEMPTY, True, request.tool, 1,
            )
        engine = agent.IncrementalConstraintEngine(planner)
        budget = next(c for c in engine.constraints if c.kind == "budget_lower_bound")
        result = engine._budget_lower_bound(budget)
        self.assertEqual(result.truth, agent.UNSAT)
        self.assertEqual(result.lower_bound, 850)
        self.assertEqual(result.certificate.kind, "BUDGET_LOWER_BOUND")

    def test_query_error_cannot_complete_a_domain(self):
        state = agent.TravelState(self.minimal_ir())
        request = agent.AcquisitionRequest("query_train_info", "test", {}, "outbound:train")
        state.query_outcomes[request.key] = agent.QueryOutcome(
            agent.QUERY_ERROR, False, request.tool, 0, "protocol error",
        )
        self.assertIsNone(state.complete_evidence_keys([request]))

    def test_nonempty_uncanonicalized_payload_is_parse_error_not_empty(self):
        state = agent.TravelState(self.minimal_ir())
        request = agent.AcquisitionRequest("query_hotel_info", "test", {}, "hotels:test")
        state.record_query_outcome(
            request, {"raw": "unexpected successful protocol text"}, [], True,
        )
        outcome = state.query_outcomes[request.key]
        self.assertEqual(outcome.status, agent.QUERY_PARSE_ERROR)
        self.assertFalse(outcome.exhaustive)
        self.assertIsNone(state.complete_evidence_keys([request]))

    def test_explicit_not_found_payload_is_certified_empty(self):
        state = agent.TravelState(self.minimal_ir())
        request = agent.AcquisitionRequest(
            "query_attraction_details", "test", {}, "details:attraction:missing",
        )
        state.record_query_outcome(
            request, {"raw": "未找到景点 missing 的详细信息"}, [], True,
        )
        outcome = state.query_outcomes[request.key]
        self.assertEqual(outcome.status, agent.QUERY_SUCCESS_EMPTY)
        self.assertTrue(outcome.exhaustive)

    def test_certificate_verifier_rejects_error_evidence(self):
        planner = object.__new__(agent.SolverInformedTravelAgent)
        planner.state = agent.TravelState(self.minimal_ir())
        key = "outbound:train"
        certificate = agent.InfeasibilityCertificate(
            "EXHAUSTIVE_NO_WITNESS", ("transport.outbound.coverage",),
            (key,), ("transport.outbound",), explanation="test",
        )
        constraint = agent.ConstraintIR(
            "transport.outbound.coverage", "candidate_coverage", "transport", True,
        )

        class StaticEngine:
            constraints = [constraint]

            @staticmethod
            def _evaluate(_constraint):
                return agent.ConstraintEvaluation(
                    constraint, agent.UNSAT, certificate=certificate,
                )

        planner.constraint_engine = StaticEngine()
        planner.state.query_outcomes[key] = agent.QueryOutcome(
            agent.QUERY_ERROR, False, "query_train_info", error="protocol error",
        )
        self.assertFalse(planner._verify_infeasibility_certificate(certificate))
        planner.state.query_outcomes[key] = agent.QueryOutcome(
            agent.QUERY_SUCCESS_EMPTY, True, "query_train_info",
        )
        self.assertTrue(planner._verify_infeasibility_certificate(certificate))

        soft_worded = agent.ConstraintIR(
            "transport.outbound.coverage", "candidate_coverage", "transport", True,
            source_text="最好有直达车",
        )
        planner.constraint_engine.constraints = [soft_worded]
        self.assertFalse(planner._verify_infeasibility_certificate(certificate))

    def test_validator_drops_only_wholly_empty_constraint_placeholders(self):
        ir = self.minimal_ir()
        ir["hotel"]["constraints"] = [
            {"field": "", "op": "", "value": None, "source_text": ""},
        ]
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        self.assertEqual(ir["hotel"]["constraints"], [])

        ir = self.minimal_ir()
        ir["hotel"]["constraints"] = [
            {"field": "services", "op": "", "value": "免费停车", "source_text": "提供免费停车"},
        ]
        agent.SolverInformedTravelAgent._validate_problem_ir(ir)
        self.assertEqual(ir["hotel"]["constraints"][0]["op"], "contains")

        ir = self.minimal_ir()
        ir["hotel"]["constraints"] = [
            {"field": "price", "op": "", "value": 900, "source_text": "每晚900元"},
        ]
        with self.assertRaisesRegex(ValueError, "unsupported constraint operator"):
            agent.SolverInformedTravelAgent._validate_problem_ir(ir)


class OfficialAdapterTests(unittest.TestCase):
    def request(self, tool, hint):
        return agent.AcquisitionRequest(tool, "test", hint, "test-key")

    def test_train_and_hotel_arguments(self):
        train = self.request("query_train_info", {"origin": "杭州", "destination": "绍兴",
                                                  "date": "2025-11-12", "seat_class": "一等座"})
        self.assertEqual(agent.OfficialToolAdapters.build(train, None)["seatClassName"], "一等座")
        hotel = self.request("query_hotel_info", {"city": "绍兴", "start_date": "2025-11-12",
                                                  "end_date": "2025-11-13", "hotel_star": 3.0})
        self.assertEqual(agent.OfficialToolAdapters.build(hotel, None)["hotelStar"], "3")

    def test_restaurant_requires_official_coordinates(self):
        request = self.request("recommend_restaurants", {"city": "绍兴"})
        with self.assertRaises(ValueError):
            agent.OfficialToolAdapters.build(request, None)


class ConflictReflectionTests(unittest.TestCase):
    def planner(self):
        planner = object.__new__(agent.SolverInformedTravelAgent)
        planner.problem_ir = {
            "trip": {"destination": "北京"},
            "attractions": {"required_names": []},
            "restaurants": {"requirements": [{}]},
        }
        planner.state = agent.TravelState(planner.problem_ir)
        planner.state.add_record(agent.EvidenceRecord(
            "anchor", "attraction", "test", {},
            {"name": "故宫博物院", "lat": 39.9, "lon": 116.4},
        ))
        class Tools:
            @staticmethod
            def has(name):
                return name in {"recommend_restaurants", "recommend_attractions"}

        planner.tools = Tools()
        return planner

    def test_recall_intents_compile_only_to_grounded_official_requests(self):
        planner = self.planner()
        requests = planner._compile_recall_expansions([
            {"constraint_id": "restaurants.requirement.0", "kind": "additional_anchor",
             "values": ["故宫博物院", "模型虚构地点"]},
            {"constraint_id": "attractions.coverage", "kind": "attraction_type",
             "values": ["历史文化", "模型自造类型"]},
        ], {"restaurants.requirement.0", "attractions.coverage"})
        self.assertEqual({request.tool for request in requests},
                         {"recommend_restaurants", "recommend_attractions"})
        self.assertTrue(all("模型" not in request.key for request in requests))

    def test_recall_compiler_is_pure_and_controller_owns_duplicate_policy(self):
        planner = self.planner()
        expansion = [{
            "constraint_id": "restaurants.requirement.0", "kind": "additional_anchor",
            "values": ["故宫博物院"],
        }]
        first = planner._compile_recall_expansions(expansion, {"restaurants.requirement.0"})
        second = planner._compile_recall_expansions(expansion, {"restaurants.requirement.0"})
        self.assertEqual(len(first), 1)
        self.assertEqual([request.key for request in second], [request.key for request in first])

    def test_reflection_compiles_only_novel_grounded_retrieval(self):
        planner = self.planner()
        planner.enable_reflection = True
        planner.max_reflections = 2
        planner._reflection_signatures = set()
        planner.semantic_events = []
        planner.metrics = {
            "reflection_calls": 0, "reflection_actions_accepted": 0,
            "reflection_duplicate_actions": 0, "recovery_rejected": 0,
            "solver_rounds": 4, "master_assignments": 3,
            "frontier_states": 6, "tool_calls": 20,
        }
        planner.constraint_engine = type("Engine", (), {"constraints": [
            agent.ConstraintIR(
                "restaurants.requirement.0", "restaurant_requirement", "restaurant",
            ),
        ]})()

        def reflect(*_args, **_kwargs):
            planner.metrics["reflection_calls"] += 1
            return {
                "diagnosis": "COVERAGE_GAP", "action": "EXPAND_RECALL",
                "reason": "restaurant anchors are under-covered",
                "recall_expansions": [{
                    "constraint_id": "restaurants.requirement.0",
                    "kind": "additional_anchor", "values": ["故宫博物院", "虚构地点"],
                }],
            }

        planner._semantic_ir_call = reflect
        result = agent.SolveResult(
            agent.SEARCH_EXHAUSTED, "stagnated",
            conflict={"kind": "no_restaurant_candidates"},
        )
        action, requests = planner._reflect_on_search("故宫附近吃饭", result, None)
        self.assertEqual(action, "EXPANDED")
        self.assertEqual(len(requests), 1)
        self.assertIn("故宫博物院", requests[0].args_hint["nearby_location"])
        second_action, second_requests = planner._reflect_on_search(
            "故宫附近吃饭", result, None,
        )
        self.assertEqual((second_action, second_requests), ("KEEP", []))
        self.assertEqual(planner.metrics["reflection_duplicate_actions"], 1)


class DeterministicSchedulerTests(unittest.TestCase):
    def planner(self, closing_time=800):
        planner = object.__new__(agent.SolverInformedTravelAgent)
        planner.problem_ir = {"trip": {"travelers": 1}}
        planner.state = agent.TravelState(planner.problem_ir)
        planner.state.add_record(agent.EvidenceRecord(
            "A", "attraction", "test", {},
            {"name": "A", "price": 0, "opening_time": 600,
             "closing_time": closing_time, "min_duration": 60},
        ))
        for rid, origin, destination, duration in (
            ("h-a", "Hotel", "A", 30), ("a-h", "A", "Hotel", 20),
        ):
            planner.state.add_record(agent.EvidenceRecord(
                rid, "route", "test", {},
                {"name": f"{origin}->{destination}", "origin": origin,
                 "destination": destination, "duration": duration, "price": 5},
            ))
        return planner

    def test_earliest_chain_waits_for_opening_time(self):
        planner = self.planner()
        template = agent.DayTemplate(
            1, "Hotel", "Hotel", 540, 800,
            [agent.ActivitySlot("d1:a1", 1, "attraction", ["A"])],
        )
        result = planner._schedule_day(template, set(), set())
        self.assertEqual(result.status, agent.READY)
        self.assertEqual(result.solution["activities"][0]["start"], 600)
        self.assertEqual(result.solution["activities"][0]["end"], 660)

    def test_earliest_chain_proves_closing_time_infeasibility(self):
        planner = self.planner(closing_time=650)
        template = agent.DayTemplate(
            1, "Hotel", "Hotel", 540, 800,
            [agent.ActivitySlot("d1:a1", 1, "attraction", ["A"])],
        )
        result = planner._schedule_day(template, set(), set())
        self.assertEqual(result.status, agent.INFEASIBLE)
        self.assertEqual(result.conflict["kind"], "temporal_infeasible")


class BestEffortIncumbentTests(unittest.TestCase):
    def test_unverified_incumbent_is_delivered_without_becoming_success(self):
        planner = object.__new__(agent.SolverInformedTravelAgent)
        planner.model = "offline"
        planner.max_solver_rounds = 1
        planner.max_reflections = 0
        planner.max_llm_calls = 10
        planner.max_tool_calls = 10
        planner.enable_reflection = False
        planner._assignment_nogoods = {}
        planner._day_nogoods = {}
        planner.last_propagation = None
        problem = ConstraintEngineTests().minimal_ir()
        planner._parse_problem_ir = lambda _query: agent.copy.deepcopy(problem)
        planner._audit_problem_ir = lambda _query, ir: ir
        planner._initial_requests = lambda: []
        planner._run_acquisition_queue = lambda *_args, **_kwargs: False
        planner._propagate_constraints = lambda *_args, **_kwargs: None
        solution = {
            "outbound_transport": "out", "return_transport": "back", "hotel": "hotel",
            "days": [{"activities": [{"record_id": "candidate"}]}],
        }
        planner._attempt_solve = lambda: agent.SolveResult(agent.READY, solution=solution)
        planner._verify = lambda _solution: {
            "success": False, "errors": [{"coverage_gap": "candidate"}],
            "evidence_coverage": 0.8, "budget_total": 400, "budget_max": 500,
        }
        planner._expansion_requests = lambda _conflict: []
        planner._reflect_on_search = lambda *_args, **_kwargs: ("KEEP", [])
        planner.to_converted_plan = lambda _solution: {
            "daily_plans": [], "budget_summary": {"currency": "CNY"},
        }
        planner._render_plan = lambda _converted: "<plan>best effort</plan>"

        plan, diagnostic = planner.run("offline test")
        self.assertTrue(plan)
        self.assertFalse(diagnostic["verification"]["success"])
        self.assertTrue(diagnostic["metrics"]["best_effort_returned"])
        self.assertFalse(diagnostic["metrics"]["verification_success"])


@unittest.skipUnless(agent.HAS_GUROBI, "Gurobi unavailable")
class GlobalAssignmentTests(unittest.TestCase):
    def test_entity_is_reserved_across_days(self):
        planner = object.__new__(agent.SolverInformedTravelAgent)
        planner.problem_ir = {"trip": {"travelers": 1, "budget_max": 100},
                              "attractions": {"constraints": []}, "restaurants": {"requirements": []}}
        planner.state = agent.TravelState(planner.problem_ir)
        planner.metrics = {"solver_runtime": 0, "master_assignments": 0}
        planner._assignment_nogoods = {}
        planner._day_nogoods = {}
        hotel = agent.EvidenceRecord("hotel", "hotel", "test", {},
                                     {"name": "Hotel", "price": 0, "lat": 30.0, "lon": 120.0})
        planner.state.add_record(hotel)
        for name in ("A", "B"):
            rec = agent.EvidenceRecord(name, "attraction", "test", {},
                                       {"name": name, "price": 1, "min_duration": 60,
                                        "opening_time": 0, "closing_time": 1440,
                                        "lat": 30.0, "lon": 120.0})
            planner.state.add_record(rec)
        days = [agent.DayTemplate(day, "Hotel", "Hotel", 480, 1200,
                                  [agent.ActivitySlot(f"d{day}:a1", day, "attraction", ["A", "B"])])
                for day in (1, 2)]
        result = planner._master_assign(days, ("out", "in", "hotel"), 0)
        self.assertEqual(result.status, agent.READY)
        self.assertEqual(set(result.solution["assignment"]), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
