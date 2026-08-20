"""Tests for the ``icd-mappings`` backend of ``pyhealth.medcode``.

These run entirely offline: the ``icd-mappings`` package ships its data inside
its wheel, so every assertion here exercises the real mapping rather than a
mock. The one thing deliberately never touched is PyHealth's own resource
server, which would make the suite depend on the network.

Author: John Wu (johnwu3)
"""

import logging
import unittest

from pyhealth.medcode import ICD9CM, ICD10CM, CrossMap, FlatMap, InnerMap
from pyhealth.medcode.cross_map import (
    BACKEND_ICDMAPPINGS,
    BACKEND_PYHEALTH,
)
from pyhealth.medcode.icd_mappings import (
    ICD_MAPPINGS_OPTIONAL_PAIRS,
    ICD_MAPPINGS_PAIRS,
    PYHEALTH_NATIVE_PAIRS,
    available_icd_mapping_pairs,
)

_LOSSY_WARNING_LOGGER = "pyhealth.medcode.cross_map"
_PREVIOUS_LEVEL = logging.NOTSET


def setUpModule():
    """Quiets the intentional lossy-pair warning for this module only.

    Never use `logging.disable()` here: it is process-global and would switch
    off warnings for every other test module in the same run.
    """
    global _PREVIOUS_LEVEL
    logger = logging.getLogger(_LOSSY_WARNING_LOGGER)
    _PREVIOUS_LEVEL = logger.level
    logger.setLevel(logging.ERROR)


def tearDownModule():
    logging.getLogger(_LOSSY_WARNING_LOGGER).setLevel(_PREVIOUS_LEVEL)


class TestBackendResolution(unittest.TestCase):
    """`_resolve_backend` is a pure function, so these need no data at all."""

    def test_native_pairs_always_prefer_pyhealth(self):
        for pair in PYHEALTH_NATIVE_PAIRS:
            self.assertEqual(
                CrossMap._resolve_backend(pair[0], pair[1], "auto"),
                BACKEND_PYHEALTH,
                msg=f"{pair} should resolve to PyHealth's own tables",
            )

    def test_overlapping_pair_stays_on_pyhealth(self):
        # ICD9CM->CCSCM is the one pair both backends could serve. It must
        # keep using PyHealth's table so existing pipelines are unaffected.
        self.assertIn(("ICD9CM", "CCSCM"), PYHEALTH_NATIVE_PAIRS)
        self.assertEqual(
            CrossMap._resolve_backend("ICD9CM", "CCSCM", "auto"),
            BACKEND_PYHEALTH,
            msg="backward compatibility for ICD9CM->CCSCM failed",
        )

    def test_gap_pairs_use_icd_mappings(self):
        for pair in ICD_MAPPINGS_PAIRS:
            if pair in PYHEALTH_NATIVE_PAIRS:
                continue
            self.assertEqual(
                CrossMap._resolve_backend(pair[0], pair[1], "auto"),
                BACKEND_ICDMAPPINGS,
                msg=f"{pair} should fall back to icd-mappings",
            )

    def test_unsupported_pair_raises_value_error(self):
        with self.assertRaises(ValueError):
            CrossMap._resolve_backend("ICD9CM", "RxNorm", "auto")

    def test_forcing_icd_mappings_on_unsupported_pair_raises(self):
        with self.assertRaises(ValueError):
            CrossMap._resolve_backend("NDC", "ATC", BACKEND_ICDMAPPINGS)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            CrossMap._resolve_backend("ICD9CM", "CCSCM", "sqlite")

    def test_available_pairs_are_registered_vocabularies(self):
        from pyhealth import medcode

        for source, target in available_icd_mapping_pairs():
            self.assertTrue(hasattr(medcode, source), msg=f"{source} missing")
            self.assertTrue(hasattr(medcode, target), msg=f"{target} missing")


class TestICDTranslation(unittest.TestCase):
    """ICD-9 <-> ICD-10, the gap PyHealth could not serve before."""

    @classmethod
    def setUpClass(cls):
        cls.to10 = CrossMap.load("ICD9CM", "ICD10CM")
        cls.to9 = CrossMap.load("ICD10CM", "ICD9CM")

    def test_forward_translation(self):
        self.assertEqual(self.to10.map("428.0"), ["I50.9"])

    def test_reverse_translation(self):
        self.assertEqual(self.to9.map("I50.9"), ["428.9"])

    def test_input_is_dot_agnostic(self):
        self.assertEqual(self.to10.map("4280"), self.to10.map("428.0"))

    def test_output_is_dotted(self):
        # icd-mappings returns undotted codes; PyHealth's convention is dotted.
        for code in self.to10.map("4280"):
            self.assertIn(".", code, msg=f"{code} was not re-dotted")

    def test_round_trip_is_lossy_and_says_so(self):
        # 428.0 (CHF unspecified) comes back as 428.9 (HF unspecified). This
        # is documented behavior, pinned here so a change is deliberate.
        there = self.to10.map("428.0")
        back = self.to9.map(there[0])
        self.assertNotEqual(back, ["428.0"])

    def test_unmapped_code_returns_empty_and_is_recorded(self):
        mapping = CrossMap.load("ICD9CM", "ICD10CM")
        self.assertEqual(mapping.map("999.99"), [])
        self.assertIn("999.99", mapping.unmapped_codes)

    def test_native_backend_exposes_empty_unmapped_codes(self):
        # The attribute is uniform across backends so callers need no branch.
        mapping = CrossMap.load("ICD10CM", "CCSR")
        self.assertEqual(mapping.unmapped_codes, set())


class TestBackendSelectionEndToEnd(unittest.TestCase):
    """The `backend` argument must survive the trip through load()."""

    def test_load_forwards_backend_argument(self):
        mapping = CrossMap.load(
            "ICD9CM", "ICD10CM", backend=BACKEND_ICDMAPPINGS
        )
        self.assertEqual(mapping.backend, BACKEND_ICDMAPPINGS)
        self.assertEqual(mapping.map("428.0"), ["I50.9"])

    def test_load_rejects_bad_backend(self):
        with self.assertRaises(ValueError):
            CrossMap.load("ICD9CM", "ICD10CM", backend="postgres")

    def test_auto_reports_which_source_it_used(self):
        self.assertEqual(
            CrossMap.load("ICD10CM", "CCSR").backend, BACKEND_ICDMAPPINGS
        )

    def test_forcing_pyhealth_on_a_gap_pair_is_allowed_but_will_not_resolve(self):
        # Explicitly asking for PyHealth's table on a pair it has none for is
        # a legitimate request; it just cannot be satisfied from the CSVs.
        self.assertEqual(
            CrossMap._resolve_backend("ICD9CM", "ICD10CM", BACKEND_PYHEALTH),
            BACKEND_PYHEALTH,
        )


class TestOptInOverlap(unittest.TestCase):
    """ICD9CM->CCSCM is servable by both; the default must never change."""

    def test_auto_never_selects_an_optional_pair(self):
        for pair in ICD_MAPPINGS_OPTIONAL_PAIRS:
            self.assertEqual(
                CrossMap._resolve_backend(pair[0], pair[1], "auto"),
                BACKEND_PYHEALTH,
                msg=f"auto must not reroute {pair} away from PyHealth",
            )

    def test_optional_pairs_are_not_in_the_auto_registry(self):
        # Structural guarantee: nothing in the auto path can reach them.
        self.assertFalse(set(ICD_MAPPINGS_OPTIONAL_PAIRS) & set(ICD_MAPPINGS_PAIRS))

    def test_forcing_icdmappings_on_an_optional_pair_is_allowed(self):
        mapping = CrossMap.load(
            "ICD9CM", "CCSCM", backend=BACKEND_ICDMAPPINGS
        )
        self.assertEqual(mapping.backend, BACKEND_ICDMAPPINGS)
        # Offline CCS, agreeing with PyHealth's hosted table.
        self.assertEqual(mapping.map("428.0"), ["108"])


class TestLossDisclosure(unittest.TestCase):
    """Mapping loss must be announced, not silently absorbed."""

    def test_lossy_pair_warns_once_at_construction(self):
        with self.assertLogs("pyhealth.medcode.cross_map", level="WARNING") as cm:
            CrossMap.load("ICD9CM", "ICD10CM")
        self.assertEqual(len(cm.output), 1, msg="expected exactly one warning")
        self.assertIn("ICD9CM", cm.output[0])

    def test_non_lossy_pair_does_not_warn(self):
        logger = logging.getLogger("pyhealth.medcode.cross_map")
        with self.assertNoLogs(logger, level="WARNING"):
            CrossMap.load("ICD10CM", "CCSR")

    def test_sparse_grouper_returns_empty_list(self):
        # Most diagnoses are not complex chronic conditions.
        mapping = CrossMap.load("ICD10CM", "CCC")
        self.assertEqual(mapping.map("J18.9"), [])
        self.assertIn("J18.9", mapping.unmapped_codes)


class TestStandardizeRoundTrip(unittest.TestCase):
    """`undot(V.standardize(x)) == x` -- the adapter's core invariant."""

    def test_icd9cm_round_trip(self):
        for code in ["4280", "25000", "486", "V3000", "E9331", "250", "8010"]:
            self.assertEqual(
                ICD9CM.standardize(code).replace(".", ""),
                code,
                msg=f"ICD9CM round trip failed for {code}",
            )

    def test_icd10cm_round_trip(self):
        for code in ["I509", "E119", "J189", "S72001A", "A00", "Z00121"]:
            self.assertEqual(
                ICD10CM.standardize(code).replace(".", ""),
                code,
                msg=f"ICD10CM round trip failed for {code}",
            )


class TestGrouperVocabularies(unittest.TestCase):
    """One real case per new vocabulary, captured from actual output."""

    def test_ccsr(self):
        self.assertEqual(CrossMap.load("ICD10CM", "CCSR").map("I50.9"), ["CIR019"])

    def test_chronic_indicators_are_normalized_to_codes(self):
        # Upstream returns a bool; a vocabulary must yield strings.
        self.assertEqual(CrossMap.load("ICD9CM", "CCI").map("428.0"), ["1"])
        self.assertEqual(CrossMap.load("ICD10CM", "CCIR").map("I50.9"), ["1"])
        self.assertEqual(CrossMap.load("ICD9CM", "CCI").map("486"), ["0"])

    def test_icd9_chapter(self):
        self.assertEqual(CrossMap.load("ICD9CM", "ICD9CHAPTER").map("428.0"), ["7"])

    def test_icd10_chapter_drops_the_label(self):
        # Upstream yields "I00-I99 | Diseases of the circulatory system".
        self.assertEqual(
            CrossMap.load("ICD10CM", "ICD10CHAPTER").map("I50.9"), ["I00-I99"]
        )

    def test_icd10_block_drops_the_label(self):
        self.assertEqual(
            CrossMap.load("ICD10CM", "ICD10BLOCK").map("I50.9"), ["I30-I5A"]
        )

    def test_complex_chronic_condition(self):
        self.assertEqual(CrossMap.load("ICD10CM", "CCC").map("I50.9"), ["cvd"])

    def test_complex_chronic_subcategory_is_trimmed(self):
        # Upstream value carries a trailing space.
        self.assertEqual(
            CrossMap.load("ICD10CM", "CCCSUB").map("I50.9"),
            ["Other Cardiovascular"],
        )


class TestFlatMap(unittest.TestCase):
    """Grouper vocabularies have no ontology, and say so honestly."""

    def test_flat_map_load(self):
        self.assertEqual(FlatMap.load("CCSR").vocabulary, "CCSR")

    def test_inner_map_load_also_works(self):
        # InnerMap.load passes refresh_cache, which FlatMap accepts+ignores.
        self.assertEqual(InnerMap.load("ICD10BLOCK").vocabulary, "ICD10BLOCK")

    def test_graph_methods_are_absent(self):
        ccsr = FlatMap.load("CCSR")
        for method in ["lookup", "get_ancestors", "get_descendants", "stat"]:
            with self.assertRaises(AttributeError, msg=f"{method} should not exist"):
                getattr(ccsr, method)("CIR019")


class TestOffline(unittest.TestCase):
    """The backend must never reach PyHealth's resource server."""

    def test_mapping_works_with_downloads_disabled(self):
        from pyhealth.medcode import inner_map, utils

        def explode(*args, **kwargs):
            raise AssertionError("attempted a network download")

        originals = (utils.download_and_read_csv, inner_map.download_and_read_csv)
        utils.download_and_read_csv = explode
        inner_map.download_and_read_csv = explode
        try:
            self.assertEqual(
                CrossMap.load("ICD10CM", "CCSR").map("I50.9"), ["CIR019"]
            )
            self.assertEqual(
                CrossMap.load("ICD9CM", "ICD10CM").map("428.0"), ["I50.9"]
            )
        finally:
            utils.download_and_read_csv = originals[0]
            inner_map.download_and_read_csv = originals[1]

    def test_vocabulary_classes_are_not_instantiated(self):
        mapping = CrossMap.load("ICD10CM", "CCSR")
        # Bound as classes, not instances -- instantiating would download.
        self.assertTrue(isinstance(mapping.s_class, type))
        self.assertTrue(isinstance(mapping.t_class, type))


class TestSequenceProcessorIntegration(unittest.TestCase):
    """The only generic consumer of CrossMap must work end to end."""

    def test_processor_maps_icd10_to_ccsr(self):
        from pyhealth.processors.sequence_processor import SequenceProcessor

        samples = [{"conditions": ["I50.9", "E11.9"]}, {"conditions": ["J18.9"]}]
        processor = SequenceProcessor(code_mapping=("ICD10CM", "CCSR"))
        processor.fit(samples, "conditions")

        # Vocabulary is CCSR categories, not raw ICD-10 codes.
        self.assertIn("CIR019", processor.code_vocab)
        self.assertNotIn("I50.9", processor.code_vocab)

        tensor = processor.process(["I50.9"])
        self.assertEqual(tensor.tolist(), [processor.code_vocab["CIR019"]])


if __name__ == "__main__":
    unittest.main()
