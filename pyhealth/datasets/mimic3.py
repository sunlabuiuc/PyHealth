import dataclasses
import logging
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MIMIC3Dataset(BaseDataset):
    """
    A dataset class for handling MIMIC-III data.

    This class is responsible for loading and managing the MIMIC-III dataset,
    which includes tables such as patients, admissions, and icustays.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the MIMIC4Dataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "mimic3".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3.yaml"
        default_tables = ["patients", "admissions", "icustays"]
        tables = default_tables + tables
        if "prescriptions" in tables:
            warnings.warn(
                "Events from prescriptions table only have date timestamp (no specific time). "
                "This may affect temporal ordering of events.",
                UserWarning,
            )
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic3",
            config_path=config_path,
            **kwargs
        )
        return

    def preprocess_noteevents(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Table-specific preprocess function which will be called by BaseDataset.load_table().
    
        Preprocesses the noteevents table by ensuring that the charttime column
        is populated. If charttime is null, it uses chartdate with a default
        time of 00:00:00.

        See: https://mimic.mit.edu/docs/iii/tables/noteevents/#chartdate-charttime-storetime.

        Args:
            df (pl.LazyFrame): The input dataframe containing noteevents data.

        Returns:
            pl.LazyFrame: The processed dataframe with updated charttime
            values.
        """
        df = df.with_columns(
            pl.when(pl.col("charttime").is_null())
            .then(pl.col("chartdate") + pl.lit(" 00:00:00"))
            .otherwise(pl.col("charttime"))
            .alias("charttime")
        )
        return df


MIMIC3_MASK_TOKEN_PATTERN = re.compile(r"\[\*\*(.*?)\*\*\]")
class MIMIC3NursingNotesDataset(BaseDataset):

    """
    A dataset class for handling the de-identified nursing notes corpus from MIMIC-III.

    This dataset loads and processes the 'id.text' (original text) and 'id.res'
    (masked text) files containing nursing notes. It aligns the masked text
    with the original text to identify the spans and labels of the masked
    sensitive information.

    This class inherits from BaseDataset but overrides the `load_data` method
    to handle the specific non-tabular format of the nursing notes files,
    unlike the standard CSV tables in other MIMIC-III datasets.

    Attributes:
        root (str): The root directory where the nursing notes files ('id.text', 'id.res') are stored.
        notes_filename (str): The name of the file containing the original nursing notes text (default: "id.text").
        notes_masked_filename (str): The name of the file containing the masked nursing notes text (default: "id.res").

        text_records (List[str]): A list of original text records after processing and filtering.
        res_records (List[str]): A list of masked text records after processing and filtering.
        masks (List[List[MaskInfo]]): A list where each element corresponds to a record
                                      and contains a list of `MaskInfo` objects detailing
                                      the identified masks and their corresponding original text spans.
    """
    records: List["ProcessedRecord"]

    def __init__(
            self,
            root: str,
            config_path: Optional[str] = None,
            notes_filename: str = "id.text",
            notes_masked_filename: str = "id.res",
            **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3_note.yaml"
        self.root = root
        self.notes_filename = notes_filename
        self.notes_masked_filename = notes_masked_filename
        self.load_data()


    # Override load_table to handle the specific text file format
    # This method is necessary because the nursing notes file is not a standard CSV
    def load_data(self) -> pl.LazyFrame:
        res_file_path = os.path.join(self.root, "id.res")
        text_file_path = os.path.join(self.root, "id.text")

        print(f"Reading records from {res_file_path} and {text_file_path}...")
        res_records = self.read_and_split_records(res_file_path)
        text_records = self.read_and_split_records(text_file_path)

        matched_res_records, matched_text_records, matched_mask_results = self.process_and_filter_records(res_records, text_records)

        self.text_records = matched_text_records
        self.res_records = matched_res_records
        self.masks = matched_mask_results

        self.records = []
        # Ensure the lists have the same length before zipping
        if not (len(matched_res_records) == len(matched_text_records) == len(matched_mask_results)):
            logger.error("Internal error: Mismatched lengths after processing and filtering. Cannot create ProcessedRecord list.")
            self.processed_records = [] # Clear in case of error
            return


        for res_rec, text_rec, mask_list in zip(matched_res_records, matched_text_records, matched_mask_results):
            self.records.append(ProcessedRecord(
                res_record=res_rec,
                text_record=text_rec,
                mask_info=mask_list
            ))

    def read_and_split_records(self, file_path):
        """
        Reads file content and splits it into records.

        Args:
            file_path (str): The path to the file.

        Returns:
            list: A list containing all records, with START_OF_RECORD and END_OF_RECORD markers removed.
                  Returns an empty list if the file does not exist or an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Use non-capturing group (?:...) for the split pattern
                records = re.split(r"START_OF_RECORD=\d+\|\|\|\|\d+\|\|\|\|", content)
                cleaned_records = []
                for record in records:
                    if record.strip():
                        # Remove the trailing END_OF_RECORD marker
                        cleaned_record = re.sub(r"\|\|\|\|END_OF_RECORD", "", record).strip()
                        cleaned_records.append(cleaned_record)
                return cleaned_records
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return []
        except Exception as e:
            print(f"An error occurred during file reading: {e}")
            return []

    def process_and_filter_records(self, res_records, text_records):
        """
        Processes pairs of res and text records, performs mapping, and filters
        records where not all masks are successfully mapped.

        Args:
            res_records (list): List of records from the .res file.
            text_records (list): List of records from the .text file.

        Returns:
            tuple: (matched_res_records, matched_text_records, matched_mask_results)
                   Lists containing records and the list of MaskResult objects that passed the filtering.
        """
        matched_res_records = []
        matched_text_records = []
        matched_mask_results = [] # This will store lists of MaskResult objects

        if len(res_records) != len(text_records):
            print("Error: Number of records in res and text files do not match. Cannot process.")
            return [], [], []

        for i in range(len(res_records)):
            res_record = res_records[i]
            text_record = text_records[i]

            # Skip empty records
            if not res_record.strip() or not text_record.strip():
                print(f"Warning: Skipping empty record {i+1}.")
                continue

            processor = MIMIC3NoteMatcher(res_record, text_record)
            mask_results = processor.masks # Get the list of MaskResult objects

            # Filtering Logic: A record is matched if all masks found in the res_record
            # are present in the mask_results with valid (non-None, non-empty text) mappings.
            all_masks_in_record = re.findall(MIMIC3_MASK_TOKEN_PATTERN, res_record)
            unique_mask_contents = set(all_masks_in_record)
            total_unique_masks = len(unique_mask_contents)

            # Create a set of mask labels that were successfully mapped to non-empty text
            successfully_mapped_labels = {
                result.label for result in mask_results if result.text is not None and result.text.strip() != ""
            }

            # Check if all unique mask contents from the res_record have a corresponding
            # successfully mapped label in the mask_results.
            # Note: This assumes the label in MaskResult is the content inside the mask [**label**].
            is_matched = False
            if total_unique_masks == 0 and not mask_results:
                # Case: No masks in res_record, and mask_results is empty (correctly)
                is_matched = True
            elif total_unique_masks > 0:
                # Check if all unique mask contents from res_record are present as labels
                # in the successfully mapped results.
                is_matched = all(mask_content in successfully_mapped_labels for mask_content in unique_mask_contents)


            if is_matched:
                print(f"Record {i+1} is matched ({len(successfully_mapped_labels)}/{total_unique_masks} unique masks mapped).")
                matched_res_records.append(res_record)
                matched_text_records.append(text_record)
                matched_mask_results.append(mask_results) # Append the list of MaskResult objects
            else:
                print(f"Record {i+1} is NOT fully matched (found {total_unique_masks} unique masks, {len(successfully_mapped_labels)} successfully mapped). Skipping.")

        return matched_res_records, matched_text_records, matched_mask_results


@dataclasses.dataclass
class MaskInfo:
    """Represents a single masked segment found and its corresponding text."""
    start: int          # Start index in the original text record (inclusive)
    end: int            # End index in the original text record (exclusive)
    label: str          # The content inside the mask, e.g., "First Name"
    text: str           # The matched text from the original record
    masked_text: str    # The original mask string from the res record, e.g., "[**First Name**]"

@dataclasses.dataclass
class ProcessedRecord:
    """Represents a single processed record with original text, masked text, and mask details."""
    res_record: str         # The masked text of the record
    text_record: str        # The original text of the record
    mask_info: List[MaskInfo] # A list of MaskResult objects detailing the masks applied

class RecordProcessor:
    def __init__(self, res_record: str, text_record: str):
        self.res_record = res_record
        self.text_record = text_record
        self.masks: list[MaskInfo] = self._process_raw_record()
        self._parsed_segments = None # Stores the parsed segments of res_record

    def _parse_res_segments(self):
        """
        Parses the structure of res_record, splitting it into alternating non-mask text segments and mask groups.

        Returns:
            list: A list of (segment_type, data) tuples.
                  segment_type is "non_mask" or "mask_group".
                  data is a non-mask text string or a list of (full_mask_string, mask_content) tuples.
        """
        segments = []
        current_res_pos = 0
        # Use the modified pattern to get both the full mask and the inner label
        mask_matches = list(re.finditer(MIMIC3_MASK_TOKEN_PATTERN, self.res_record))
        mask_idx = 0

        while current_res_pos < len(self.res_record):
            # Determine the start of the next mask or the end of the res_record
            next_mask_start = len(self.res_record)
            next_mask_match = None
            if mask_idx < len(mask_matches):
                m = mask_matches[mask_idx]
                if m.start() >= current_res_pos:
                    next_mask_start = m.start()
                    next_mask_match = m
                else: # Should not happen with correct regex and indexing, but safety
                    mask_idx += 1
                    continue

            if next_mask_start > current_res_pos:
                # Current position is the start of non-mask text
                non_mask_text = self.res_record[current_res_pos : next_mask_start]
                # Only non-empty non-mask segments are meaningful as anchors or content
                if non_mask_text:
                    segments.append(("non_mask", non_mask_text))
                current_res_pos = next_mask_start # Move past the non-mask segment

            # Check if the current position is now the start of a mask
            if next_mask_match and next_mask_match.start() == current_res_pos:
                # Process a mask group
                mask_group_details = [] # Stores tuples: (full_mask_string, mask_content)
                temp_pos = current_res_pos # Temporary pointer to find consecutive masks

                # Find all consecutive masks starting exactly at temp_pos
                current_mask_match_idx = mask_idx # Start checking from the current mask_idx
                while current_mask_match_idx < len(mask_matches):
                    m = mask_matches[current_mask_match_idx]
                    if m.start() == temp_pos:
                        mask_group_details.append((m.group(1), m.group(2))) # Capture full mask and label
                        temp_pos = m.end() # Temporary pointer skips the current mask's length
                        current_mask_match_idx += 1 # Advance to the next potential mask_match
                    else:
                        break # Stop if the next match is not consecutive

                segments.append(("mask_group", mask_group_details))
                current_res_pos = temp_pos # Main pointer skips the entire mask group
                mask_idx = current_mask_match_idx # Update main mask_idx to where the inner loop stopped

        return segments


    def _distribute_text_to_masks(self, mask_details_list: list[tuple[str, str]], text_segment_start_in_text: int, text_segment_end_in_text: int):
        """
        Helper: Distributes a segment of original text to a list of mask details and records results.
        mask_details_list is a list of (full_mask_string, mask_content) tuples.
        Records results directly to self._results as MaskResult objects.

        If num_masks == 1, the entire text segment is assigned to the single mask.
        If num_masks > 1 (consecutive masks), the text segment is split based on word count.
        """
        full_text_segment = self.text_record[text_segment_start_in_text : text_segment_end_in_text]
        words = full_text_segment.split()
        num_masks = len(mask_details_list)

        if not num_masks:
            if full_text_segment.strip():
                print(f"Warning: Found text segment '{full_text_segment[:50]}...' but no corresponding masks in distribution step.")
            return

        if num_masks == 1:
            # If there is only one mask, assign the entire text segment to it
            full_mask_string, mask_label = mask_details_list[0]
            # The start and end are simply the boundaries of the text segment provided
            self.masks.append(MaskInfo(
                label=mask_label,
                text=full_text_segment,
                start=text_segment_start_in_text,
                end=text_segment_end_in_text,
                masked_text=full_mask_string
            ))
        else:
            # If there are multiple consecutive masks, apply the word-based average distribution heuristic
            words_per_mask_base = len(words) // num_masks
            extra_words = len(words) % num_masks
            word_idx = 0 # Index in the 'words' list
            current_char_pos_in_segment = 0 # Character position within the full_text_segment for text assignment

            for full_mask_string, mask_label in mask_details_list:
                num_words_this_mask = words_per_mask_base + (1 if extra_words > 0 else 0)
                if extra_words > 0:
                    extra_words -= 1

                # Get the words for this mask
                assigned_text_words = words[word_idx : word_idx + num_words_this_mask]
                assigned_text = " ".join(assigned_text_words)

                # Calculate start and end indices within the original text_record
                # Find the start of the assigned text within the full_text_segment
                # Use the character position to narrow the search range for find
                search_start_in_segment = current_char_pos_in_segment
                start_in_segment = full_text_segment.find(assigned_text, search_start_in_segment)

                if start_in_segment == -1 and assigned_text.strip():
                    # Fallback/Warning if exact match not found, try to approximate start based on word index
                    print(f"Warning: Could not find assigned text '{assigned_text[:50]}...' within segment '{full_text_segment[:50]}...' for mask '{mask_label}'. Approximating position.")
                    # Re-calculate assigned_text just in case
                    assigned_text = " ".join(words[word_idx : word_idx + num_words_this_mask])
                    # Approximate start based on summing lengths of previous words and spaces
                    # Account for potential leading space if not the very first word
                    approx_start = sum(len(w) + 1 for w in words[:word_idx]) # +1 for space after each word
                    # Clamp approximation to segment bounds
                    start_in_segment = max(0, min(approx_start, len(full_text_segment)))


                # Calculate end index (exclusive)
                # If start_in_segment is -1, the assigned text wasn't found/approximated,
                # so setting end to -1 implies an invalid range.
                end_in_segment = start_in_segment + len(assigned_text) if start_in_segment != -1 else -1 # End is start + length


                # Record the result
                self.masks.append(MaskInfo(
                    label=mask_label,
                    text=assigned_text,
                    # Add the segment's start offset to get the absolute position in text_record
                    start=text_segment_start_in_text + start_in_segment if start_in_segment != -1 else -1, # Indicate failure with -1 or handle differently
                    end=text_segment_start_in_text + end_in_segment if start_in_segment != -1 else -1,
                    masked_text=full_mask_string
                ))

                # Update indices for the next mask
                word_idx += num_words_this_mask
                # Update character position based on the end of the assigned text in the segment
                if start_in_segment != -1:
                    current_char_pos_in_segment = end_in_segment
                else:
                    # If find failed, try to estimate the next character position based on word count
                    current_char_pos_in_segment = sum(len(w) + 1 for w in words[:word_idx])


    def _process_raw_record(self) -> list[MaskInfo]:
        """
        Executes the record parsing and alignment process, generating detailed results.

        Returns:
            list[MaskInfo]: A list of MaskResult objects for each identified mask.
                              Returns an empty list if alignment fails critically.
        """
        self.masks = [] # Reset results for each call
        self._parsed_segments = self._parse_res_segments()
        text_current_pos = 0 # Pointer, tracks position in text_record

        for i in range(len(self._parsed_segments)):
            segment_type, segment_data = self._parsed_segments[i]

            if segment_type == "non_mask":
                anchor = segment_data
                anchor_match_start_in_text = self.text_record.find(anchor, text_current_pos)

                if anchor_match_start_in_text == -1:
                    print(f"Warning: Anchor '{anchor[:50]}...' not found in text after pos {text_current_pos}. Alignment likely broken for this record.")
                    self.masks = [] # Clear results to indicate failure
                    return self.masks # Interrupt processing

                anchor_match_end_in_text = anchor_match_start_in_text + len(anchor)

                # If the previous segment was a mask group, the text between the previous
                # text_current_pos and the start of this anchor is the text for that group.
                if i > 0:
                    prev_segment_type, prev_segment_data = self._parsed_segments[i-1]
                    if prev_segment_type == "mask_group":
                        text_segment_start_for_group = text_current_pos
                        text_segment_end_for_group = anchor_match_start_in_text
                        # Only process if the text segment is not empty
                        if text_segment_start_for_group < text_segment_end_for_group:
                            self._distribute_text_to_masks(prev_segment_data, text_segment_start_for_group, text_segment_end_for_group)
                        elif prev_segment_data and self.text_record[text_segment_start_for_group:text_segment_end_for_group].strip():
                            # Case where text segment is empty but there were masks, could indicate a mismatch
                            print(f"Warning: Zero-length text segment found for mask group at res_pos corresponding to segment {i-1} but non-empty text expected near text_pos {text_current_pos}. Mask labels: {[lbl for _, lbl in prev_segment_data]}.")
                            # Decide how to handle: assign empty string, indicate failure, etc.
                            # For now, _distribute_text_to_masks will handle empty input gracefully.
                            self._distribute_text_to_masks(prev_segment_data, text_segment_start_for_group, text_segment_end_for_group)


                text_current_pos = anchor_match_end_in_text

            else: # segment_type == "mask_group"
                # Mask groups are processed when the *next* non-mask segment (anchor) is found,
                # or at the end of the record if the last segment is a mask group.
                pass

        # --- Handle the last segment if it's a mask group ---
        if self._parsed_segments and self._parsed_segments[-1][0] == "mask_group":
            last_mask_group_details = self._parsed_segments[-1][1]
            remaining_text_start = text_current_pos
            remaining_text_end = len(self.text_record)
            # Only process if there is text to distribute
            if remaining_text_start < remaining_text_end:
                self._distribute_text_to_masks(last_mask_group_details, remaining_text_start, remaining_text_end)
            elif last_mask_group_details and self.text_record[remaining_text_start:remaining_text_end].strip():
                # Case where remaining text segment is empty but there were masks
                print(f"Warning: Zero-length remaining text segment found for last mask group starting near text_pos {text_current_pos}. Mask labels: {[lbl for _, lbl in last_mask_group_details]}.")
                self._distribute_text_to_masks(last_mask_group_details, remaining_text_start, remaining_text_end)


        # Optional: Check for trailing text if the last segment was non_mask
        elif text_current_pos < len(self.text_record):
            trailing_text = self.text_record[text_current_pos:]
            if trailing_text.strip():
                print(f"Warning: Trailing non-empty text found in text_record but no corresponding segments in res_record: '{trailing_text[:50]}...' starting at index {text_current_pos}")
                # Decide if this constitutes an alignment failure or just unexpected extra text.

        return self.masks

    def get_processed_record(self) -> ProcessedRecord:
        """Returns the processed record as a ProcessedRecord object."""
        return ProcessedRecord(
            res_record=self.res_record,
            text_record=self.text_record,
            mask_info=self.masks
        )

class MIMIC3NoteMatcher:
    def __init__(self, res_record: str, text_record: str):
        self.res_record = res_record
        self.text_record = text_record
        self.mask_pattern = re.compile(r"(\[\*\*(.*?)\*\*\])")
        self.masks: list[MaskInfo] = self._process_raw_record()
        self._parsed_segments = None # Stores the parsed segments of res_record

    def _parse_res_segments(self):
        """
        Parses the structure of res_record, splitting it into alternating non-mask text segments and mask groups.

        Returns:
            list: A list of (segment_type, data) tuples.
                  segment_type is "non_mask" or "mask_group".
                  data is a non-mask text string or a list of (full_mask_string, mask_content) tuples.
        """
        segments = []
        current_res_pos = 0
        # Use the modified pattern to get both the full mask and the inner label
        mask_matches = list(re.finditer(self.mask_pattern, self.res_record))
        mask_idx = 0

        while current_res_pos < len(self.res_record):
            # Determine the start of the next mask or the end of the res_record
            next_mask_start = len(self.res_record)
            next_mask_match = None
            if mask_idx < len(mask_matches):
                m = mask_matches[mask_idx]
                if m.start() >= current_res_pos:
                    next_mask_start = m.start()
                    next_mask_match = m
                else: # Should not happen with correct regex and indexing, but safety
                    mask_idx += 1
                    continue

            if next_mask_start > current_res_pos:
                # Current position is the start of non-mask text
                non_mask_text = self.res_record[current_res_pos : next_mask_start]
                # Only non-empty non-mask segments are meaningful as anchors or content
                if non_mask_text:
                    segments.append(("non_mask", non_mask_text))
                current_res_pos = next_mask_start # Move past the non-mask segment

            # Check if the current position is now the start of a mask
            if next_mask_match and next_mask_match.start() == current_res_pos:
                # Process a mask group
                mask_group_details = [] # Stores tuples: (full_mask_string, mask_content)
                temp_pos = current_res_pos # Temporary pointer to find consecutive masks

                # Find all consecutive masks starting exactly at temp_pos
                current_mask_match_idx = mask_idx # Start checking from the current mask_idx
                while current_mask_match_idx < len(mask_matches):
                    m = mask_matches[current_mask_match_idx]
                    if m.start() == temp_pos:
                        mask_group_details.append((m.group(1), m.group(2))) # Capture full mask and label
                        temp_pos = m.end() # Temporary pointer skips the current mask's length
                        current_mask_match_idx += 1 # Advance to the next potential mask_match
                    else:
                        break # Stop if the next match is not consecutive

                segments.append(("mask_group", mask_group_details))
                current_res_pos = temp_pos # Main pointer skips the entire mask group
                mask_idx = current_mask_match_idx # Update main mask_idx to where the inner loop stopped

        return segments


    def _distribute_text_to_masks(self, mask_details_list: list[tuple[str, str]], text_segment_start_in_text: int, text_segment_end_in_text: int):
        """
        Helper: Distributes a segment of original text to a list of mask details and records results.
        mask_details_list is a list of (full_mask_string, mask_content) tuples.
        Records results directly to self._results as MaskResult objects.

        If num_masks == 1, the entire text segment is assigned to the single mask.
        If num_masks > 1 (consecutive masks), the text segment is split based on word count.
        """
        full_text_segment = self.text_record[text_segment_start_in_text : text_segment_end_in_text]
        words = full_text_segment.split()
        num_masks = len(mask_details_list)

        if not num_masks:
            if full_text_segment.strip():
                print(f"Warning: Found text segment '{full_text_segment[:50]}...' but no corresponding masks in distribution step.")
            return

        if num_masks == 1:
            # If there is only one mask, assign the entire text segment to it
            full_mask_string, mask_label = mask_details_list[0]
            # The start and end are simply the boundaries of the text segment provided
            self.masks.append(MaskInfo(
                label=mask_label,
                text=full_text_segment,
                start=text_segment_start_in_text,
                end=text_segment_end_in_text,
                masked_text=full_mask_string
            ))
        else:
            # If there are multiple consecutive masks, apply the word-based average distribution heuristic
            words_per_mask_base = len(words) // num_masks
            extra_words = len(words) % num_masks
            word_idx = 0 # Index in the 'words' list
            current_char_pos_in_segment = 0 # Character position within the full_text_segment for text assignment

            for full_mask_string, mask_label in mask_details_list:
                num_words_this_mask = words_per_mask_base + (1 if extra_words > 0 else 0)
                if extra_words > 0:
                    extra_words -= 1

                # Get the words for this mask
                assigned_text_words = words[word_idx : word_idx + num_words_this_mask]
                assigned_text = " ".join(assigned_text_words)

                # Calculate start and end indices within the original text_record
                # Find the start of the assigned text within the full_text_segment
                # Use the character position to narrow the search range for find
                search_start_in_segment = current_char_pos_in_segment
                start_in_segment = full_text_segment.find(assigned_text, search_start_in_segment)

                if start_in_segment == -1 and assigned_text.strip():
                    # Fallback/Warning if exact match not found, try to approximate start based on word index
                    print(f"Warning: Could not find assigned text '{assigned_text[:50]}...' within segment '{full_text_segment[:50]}...' for mask '{mask_label}'. Approximating position.")
                    # Re-calculate assigned_text just in case
                    assigned_text = " ".join(words[word_idx : word_idx + num_words_this_mask])
                    # Approximate start based on summing lengths of previous words and spaces
                    # Account for potential leading space if not the very first word
                    approx_start = sum(len(w) + 1 for w in words[:word_idx]) # +1 for space after each word
                    # Clamp approximation to segment bounds
                    start_in_segment = max(0, min(approx_start, len(full_text_segment)))


                # Calculate end index (exclusive)
                # If start_in_segment is -1, the assigned text wasn't found/approximated,
                # so setting end to -1 implies an invalid range.
                end_in_segment = start_in_segment + len(assigned_text) if start_in_segment != -1 else -1 # End is start + length


                # Record the result
                self.masks.append(MaskInfo(
                    label=mask_label,
                    text=assigned_text,
                    # Add the segment's start offset to get the absolute position in text_record
                    start=text_segment_start_in_text + start_in_segment if start_in_segment != -1 else -1, # Indicate failure with -1 or handle differently
                    end=text_segment_start_in_text + end_in_segment if start_in_segment != -1 else -1,
                    masked_text=full_mask_string
                ))

                # Update indices for the next mask
                word_idx += num_words_this_mask
                # Update character position based on the end of the assigned text in the segment
                if start_in_segment != -1:
                    current_char_pos_in_segment = end_in_segment
                else:
                    # If find failed, try to estimate the next character position based on word count
                    current_char_pos_in_segment = sum(len(w) + 1 for w in words[:word_idx])


    def _process_raw_record(self) -> list[MaskInfo]:
        """
        Executes the record parsing and alignment process, generating detailed results.

        Returns:
            list[MaskInfo]: A list of MaskResult objects for each identified mask.
                              Returns an empty list if alignment fails critically.
        """
        self.masks = [] # Reset results for each call
        self._parsed_segments = self._parse_res_segments()
        text_current_pos = 0 # Pointer, tracks position in text_record

        for i in range(len(self._parsed_segments)):
            segment_type, segment_data = self._parsed_segments[i]

            if segment_type == "non_mask":
                anchor = segment_data
                anchor_match_start_in_text = self.text_record.find(anchor, text_current_pos)

                if anchor_match_start_in_text == -1:
                    print(f"Warning: Anchor '{anchor[:50]}...' not found in text after pos {text_current_pos}. Alignment likely broken for this record.")
                    self.masks = [] # Clear results to indicate failure
                    return self.masks # Interrupt processing

                anchor_match_end_in_text = anchor_match_start_in_text + len(anchor)

                # If the previous segment was a mask group, the text between the previous
                # text_current_pos and the start of this anchor is the text for that group.
                if i > 0:
                    prev_segment_type, prev_segment_data = self._parsed_segments[i-1]
                    if prev_segment_type == "mask_group":
                        text_segment_start_for_group = text_current_pos
                        text_segment_end_for_group = anchor_match_start_in_text
                        # Only process if the text segment is not empty
                        if text_segment_start_for_group < text_segment_end_for_group:
                            self._distribute_text_to_masks(prev_segment_data, text_segment_start_for_group, text_segment_end_for_group)
                        elif prev_segment_data and self.text_record[text_segment_start_for_group:text_segment_end_for_group].strip():
                            # Case where text segment is empty but there were masks, could indicate a mismatch
                            print(f"Warning: Zero-length text segment found for mask group at res_pos corresponding to segment {i-1} but non-empty text expected near text_pos {text_current_pos}. Mask labels: {[lbl for _, lbl in prev_segment_data]}.")
                            # Decide how to handle: assign empty string, indicate failure, etc.
                            # For now, _distribute_text_to_masks will handle empty input gracefully.
                            self._distribute_text_to_masks(prev_segment_data, text_segment_start_for_group, text_segment_end_for_group)


                text_current_pos = anchor_match_end_in_text

            else: # segment_type == "mask_group"
                # Mask groups are processed when the *next* non-mask segment (anchor) is found,
                # or at the end of the record if the last segment is a mask group.
                pass

        # --- Handle the last segment if it's a mask group ---
        if self._parsed_segments and self._parsed_segments[-1][0] == "mask_group":
            last_mask_group_details = self._parsed_segments[-1][1]
            remaining_text_start = text_current_pos
            remaining_text_end = len(self.text_record)
            # Only process if there is text to distribute
            if remaining_text_start < remaining_text_end:
                self._distribute_text_to_masks(last_mask_group_details, remaining_text_start, remaining_text_end)
            elif last_mask_group_details and self.text_record[remaining_text_start:remaining_text_end].strip():
                # Case where remaining text segment is empty but there were masks
                print(f"Warning: Zero-length remaining text segment found for last mask group starting near text_pos {text_current_pos}. Mask labels: {[lbl for _, lbl in last_mask_group_details]}.")
                self._distribute_text_to_masks(last_mask_group_details, remaining_text_start, remaining_text_end)


        # Optional: Check for trailing text if the last segment was non_mask
        elif text_current_pos < len(self.text_record):
            trailing_text = self.text_record[text_current_pos:]
            if trailing_text.strip():
                print(f"Warning: Trailing non-empty text found in text_record but no corresponding segments in res_record: '{trailing_text[:50]}...' starting at index {text_current_pos}")
                # Decide if this constitutes an alignment failure or just unexpected extra text.

        return self.masks

    def get_processed_record(self) -> ProcessedRecord:
        """Returns the processed record as a ProcessedRecord object."""
        return ProcessedRecord(
            res_record=self.res_record,
            text_record=self.text_record,
            mask_info=self.masks
        )



@dataclasses.dataclass
class MaskInfo:
    """Represents a single masked segment found and its corresponding text."""
    start: int          # Start index in the original text record (inclusive)
    end: int            # End index in the original text record (exclusive)
    label: str          # The content inside the mask, e.g., "First Name"
    text: str           # The matched text from the original record
    masked_text: str    # The original mask string from the res record, e.g., "[**First Name**]"

@dataclasses.dataclass
class ProcessedRecord:
    """Represents a single processed record with original text, masked text, and mask details."""
    res_record: str         # The masked text of the record
    text_record: str        # The original text of the record
    mask_info: List[MaskInfo] # A list of MaskResult objects detailing the masks applied
