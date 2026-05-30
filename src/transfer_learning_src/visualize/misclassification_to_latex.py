#!/usr/bin/env python3
"""
Misclassification Analysis to LaTeX Table Converter

This script converts JSON files containing misclassification analysis results
into LaTeX tables suitable for academic papers.

Usage:
    python misclassification_to_latex.py --input_dir /path/to/json/files --output_dir /path/to/output
"""

import os
import sys
import logging
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class MisclassificationToLatexConverter:
    """Converts misclassification analysis JSON files to LaTeX tables."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_compound_key(self, key: str) -> Tuple[str, str]:
        """Parse compound key to extract compound ID and protein name."""
        try:
            # Expected format: "compound_104_eeAChE-19" or "compound_104_MLCK-179"
            # Split by underscore and dash to extract parts
            parts = re.split(r'[_-]', key)
            
            if len(parts) >= 3:
                # Find compound ID (should be the number after "compound")
                compound_idx = -1
                for i, part in enumerate(parts):
                    if part == "compound" and i + 1 < len(parts):
                        compound_idx = i + 1
                        break
                
                compound_id = parts[compound_idx] if compound_idx != -1 else "Unknown"
                
                # The protein name is typically the part before the last dash and number
                # For "compound_104_eeAChE-19", protein would be "eeAChE"
                # For "compound_12_MLCK-179", protein would be "MLCK"
                protein_parts = []
                for i, part in enumerate(parts[2:], 2):  # Skip "compound" and ID
                    if not part.isdigit():  # Skip numeric parts (likely IDs)
                        protein_parts.append(part)
                    else:
                        break
                
                protein = ''.join(protein_parts) if protein_parts else "Unknown"
                
                return compound_id, protein
            else:
                # Fallback: extract numbers and assume rest is protein
                numbers = re.findall(r'\d+', key)
                compound_id = numbers[0] if numbers else "Unknown"
                
                # Extract non-numeric parts as protein
                protein_match = re.search(r'[A-Za-z]+', key)
                protein = protein_match.group() if protein_match else "Unknown"
                
                return compound_id, protein
                
        except Exception as e:
            logger.warning(f"Could not parse compound key '{key}': {e}")
            return "Unknown", "Unknown"
    
    def generate_latex_table_for_fold(self, json_file: Path) -> str:
        """Generate LaTeX table for a single fold."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Could not read {json_file}: {e}")
            return ""
        
        baseline_distance = data.get('baseline_avg_knn_dist_for_correct_samples', -1.0)
        misclassified_data = data.get('misclassified_sample_analysis', {})
        
        if not misclassified_data:
            logger.warning(f"No misclassified data found in {json_file}")
            return ""
        
        # Extract fold number from filename
        fold_match = re.search(r'fold_(\d+)', json_file.stem)
        fold_num = fold_match.group(1) if fold_match else "unknown"
        
        # Process entries
        entries = []
        for key, analysis in misclassified_data.items():
            compound_id, protein = self.parse_compound_key(key)
            
            # Determine error type
            true_label = analysis.get('true_label', 0)
            predicted_label = analysis.get('predicted_label', 0)
            
            if true_label == 0 and predicted_label == 1:
                error_type = "False Positive"
            elif true_label == 1 and predicted_label == 0:
                error_type = "False Negative"
            else:
                error_type = "Unknown"
            
            entries.append({
                'compound_id': compound_id,
                'protein': protein,
                'similarity_ratio': analysis.get('similarity_ratio'),
                'purity_score': analysis.get('neighborhood_purity_correct_class', 0),
                'outlier_score': analysis.get('outlier_score'),
                'error_type': error_type
            })
        
        # Sort entries by compound ID, then by protein
        entries.sort(key=lambda x: (int(x['compound_id']) if x['compound_id'].isdigit() else float('inf'), x['protein']))
        
        # Generate LaTeX table
        latex_lines = [
            "\\begin{table}[H]",
            "    \\centering",
            "    \\small",
            f"    \\caption{{Analysis of misclassified compounds in fold {fold_num} (baseline average kNN distance for correctly classified samples: {baseline_distance:.3f}).}}",
            f"    \\label{{tab:misclassified_compounds_{fold_num}}}",
            "    \\begin{smarttabular}{clcccc}",
            "        \\toprule",
            "        \\textbf{Compound ID} & \\textbf{Protein} & \\textbf{Similarity Ratio} & \\textbf{Purity} & \\textbf{Outlier Score} & \\textbf{Error Type} \\\\",
            "        \\midrule"
        ]
        
        # Add table rows
        for entry in entries:
            compound_id = entry['compound_id']
            protein = entry['protein']
            
            # Format similarity ratio
            if entry['similarity_ratio'] is not None:
                similarity_ratio = f"{entry['similarity_ratio']:.3f}"
            else:
                similarity_ratio = "N/A"
            
            # Format purity as percentage
            purity_percent = f"{entry['purity_score'] * 100:.0f}\\%"
            
            # Format outlier score
            if entry['outlier_score'] is not None:
                outlier_score = f"{entry['outlier_score']:.2f}"
            else:
                outlier_score = "N/A"
            
            error_type = entry['error_type']
            
            # Create table row
            row = f"        {compound_id} & {protein} & {similarity_ratio} & {purity_percent} & {outlier_score} & {error_type} \\\\"
            latex_lines.append(row)
        
        # Table footer
        latex_lines.extend([
            "        \\bottomrule",
            "    \\end{smarttabular}",
            "\\end{table}"
        ])
        
        return '\n'.join(latex_lines)
    
    def generate_combined_latex_table(self, json_files: List[Path]) -> str:
        """Generate a combined LaTeX table for all folds."""
        all_entries = []
        baseline_distances = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                baseline_dist = data.get('baseline_avg_knn_dist_for_correct_samples', -1.0)
                if baseline_dist > 0:
                    baseline_distances.append(baseline_dist)
                
                misclassified_data = data.get('misclassified_sample_analysis', {})
                
                for key, analysis in misclassified_data.items():
                    compound_id, protein = self.parse_compound_key(key)
                    
                    # Determine error type
                    true_label = analysis.get('true_label', 0)
                    predicted_label = analysis.get('predicted_label', 0)
                    
                    if true_label == 0 and predicted_label == 1:
                        error_type = "False Positive"
                    elif true_label == 1 and predicted_label == 0:
                        error_type = "False Negative"
                    else:
                        error_type = "Unknown"
                    
                    all_entries.append({
                        'compound_id': compound_id,
                        'protein': protein,
                        'similarity_ratio': analysis.get('similarity_ratio'),
                        'purity_score': analysis.get('neighborhood_purity_correct_class', 0),
                        'outlier_score': analysis.get('outlier_score'),
                        'error_type': error_type
                    })
                    
            except Exception as e:
                logger.warning(f"Could not process {json_file}: {e}")
        
        if not all_entries:
            logger.warning("No misclassified entries found for combined table.")
            return ""
        
        # Sort entries by compound ID, then by protein
        all_entries.sort(key=lambda x: (int(x['compound_id']) if x['compound_id'].isdigit() else float('inf'), x['protein']))
        
        # Calculate average baseline distance
        avg_baseline = sum(baseline_distances) / len(baseline_distances) if baseline_distances else -1.0
        
        # Generate LaTeX table
        latex_lines = [
            "\\begin{table}[H]",
            "    \\centering",
            "    \\small",
            f"    \\caption{{Analysis of misclassified compounds across all folds (average baseline kNN distance for correctly classified samples: {avg_baseline:.3f}).}}",
            "    \\label{tab:misclassified_compounds_combined}",
            "    \\begin{smarttabular}{clcccc}",
            "        \\toprule",
            "        \\textbf{Compound ID} & \\textbf{Protein} & \\textbf{Similarity Ratio} & \\textbf{Purity} & \\textbf{Outlier Score} & \\textbf{Error Type} \\\\",
            "        \\midrule"
        ]
        
        # Add table rows
        for entry in all_entries:
            compound_id = entry['compound_id']
            protein = entry['protein']
            
            # Format similarity ratio
            if entry['similarity_ratio'] is not None:
                similarity_ratio = f"{entry['similarity_ratio']:.3f}"
            else:
                similarity_ratio = "N/A"
            
            # Format purity as percentage
            purity_percent = f"{entry['purity_score'] * 100:.0f}\\%"
            
            # Format outlier score
            if entry['outlier_score'] is not None:
                outlier_score = f"{entry['outlier_score']:.2f}"
            else:
                outlier_score = "N/A"
            
            error_type = entry['error_type']
            
            # Create table row
            row = f"        {compound_id} & {protein} & {similarity_ratio} & {purity_percent} & {outlier_score} & {error_type} \\\\"
            latex_lines.append(row)
        
        # Table footer
        latex_lines.extend([
            "        \\bottomrule",
            "    \\end{smarttabular}",
            "\\end{table}"
        ])
        
        return '\n'.join(latex_lines)
    
    def parse_file_metadata(self, file_path: Path) -> Tuple[str, str]:
        """Parse threshold and affinity type from file path."""
        path_str = str(file_path)
        
        # Extract threshold (e.g., "threshold_5")
        threshold_match = re.search(r'threshold_(\d+)', path_str)
        threshold = f"threshold_{threshold_match.group(1)}" if threshold_match else "unknown_threshold"
        
        # Extract affinity type (e.g., "pic50", "pk")
        # Look for the directory name after threshold
        threshold_pattern = r'threshold_\d+[/\\]([^/\\]+)'
        affinity_match = re.search(threshold_pattern, path_str)
        affinity_type = affinity_match.group(1) if affinity_match else "unknown_affinity"
        
        return threshold, affinity_type
    
    def group_files_by_metadata(self, json_files: List[Path]) -> Dict[Tuple[str, str], List[Path]]:
        """Group files by threshold and affinity type."""
        grouped_files = {}
        
        for file_path in json_files:
            threshold, affinity_type = self.parse_file_metadata(file_path)
            key = (threshold, affinity_type)
            
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(file_path)
        
        return grouped_files

    def convert_all_files(self, pattern: str = "*advanced_misclassification_analysis.json"):
        """Convert all matching JSON files to LaTeX tables."""
        # Use rglob for recursive search instead of glob
        json_files = list(self.input_dir.rglob(pattern))
        
        if not json_files:
            logger.warning(f"No files matching pattern '{pattern}' found recursively in {self.input_dir}")
            return
        
        logger.info(f"Found {len(json_files)} JSON files to convert")
        
        # Group files by threshold and affinity type
        grouped_files = self.group_files_by_metadata(json_files)
        logger.info(f"Found {len(grouped_files)} different threshold/affinity combinations")
        
        all_individual_tables = []
        
        # Process each group separately
        for (threshold, affinity_type), group_files in grouped_files.items():
            logger.info(f"Processing {len(group_files)} files for {threshold}/{affinity_type}")

            # Output directory mirrors the image structure: output_dir/threshold_N/affinity/
            group_output_dir = self.output_dir / threshold / affinity_type
            group_output_dir.mkdir(parents=True, exist_ok=True)

            # Convert individual fold files for this group
            group_individual_tables = []
            for json_file in sorted(group_files):
                logger.info(f"Converting {json_file.name}")
                latex_content = self.generate_latex_table_for_fold(json_file)

                if latex_content:
                    output_file = group_output_dir / f"{json_file.stem}_table.tex"
                    with open(output_file, 'w') as f:
                        f.write(latex_content)
                    logger.info(f"Saved LaTeX table to {output_file}")

                    group_individual_tables.append(latex_content)
                    all_individual_tables.append(latex_content)

            # Generate combined table for this group
            if len(group_files) > 1:
                logger.info(f"Generating combined LaTeX table for {threshold}/{affinity_type}")
                combined_latex = self.generate_combined_latex_table_for_group(group_files, threshold, affinity_type)

                if combined_latex:
                    combined_output_file = group_output_dir / "combined_misclassification_table.tex"
                    with open(combined_output_file, 'w') as f:
                        f.write(combined_latex)
                    logger.info(f"Saved combined LaTeX table to {combined_output_file}")
        
        # Generate a complete document with all tables organized by groups
        if all_individual_tables:
            self.generate_complete_document_with_groups(grouped_files)
    
    def generate_combined_latex_table_for_group(self, json_files: List[Path], threshold: str, affinity_type: str) -> str:
        """Generate a combined LaTeX table for a specific threshold/affinity group."""
        all_entries = []
        baseline_distances = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                baseline_dist = data.get('baseline_avg_knn_dist_for_correct_samples', -1.0)
                if baseline_dist > 0:
                    baseline_distances.append(baseline_dist)
                
                misclassified_data = data.get('misclassified_sample_analysis', {})
                
                for key, analysis in misclassified_data.items():
                    compound_id, protein = self.parse_compound_key(key)
                    
                    # Determine error type
                    true_label = analysis.get('true_label', 0)
                    predicted_label = analysis.get('predicted_label', 0)
                    
                    if true_label == 0 and predicted_label == 1:
                        error_type = "False Positive"
                    elif true_label == 1 and predicted_label == 0:
                        error_type = "False Negative"
                    else:
                        error_type = "Unknown"
                    
                    all_entries.append({
                        'compound_id': compound_id,
                        'protein': protein,
                        'similarity_ratio': analysis.get('similarity_ratio'),
                        'purity_score': analysis.get('neighborhood_purity_correct_class', 0),
                        'outlier_score': analysis.get('outlier_score'),
                        'error_type': error_type
                    })
                    
            except Exception as e:
                logger.warning(f"Could not process {json_file}: {e}")
        
        if not all_entries:
            logger.warning(f"No misclassified entries found for {threshold}/{affinity_type} combined table.")
            return ""
        
        # Sort entries by compound ID, then by protein
        all_entries.sort(key=lambda x: (int(x['compound_id']) if x['compound_id'].isdigit() else float('inf'), x['protein']))
        
        # Calculate average baseline distance
        avg_baseline = sum(baseline_distances) / len(baseline_distances) if baseline_distances else -1.0
        
        # Generate LaTeX table
        latex_lines = [
            "\\begin{table}[H]",
            "    \\centering",
            "    \\small",
            f"    \\caption{{Analysis of misclassified compounds for {threshold.replace('_', ' ')} {affinity_type.upper()} (average baseline kNN distance for correctly classified samples: {avg_baseline:.3f}).}}",
            f"    \\label{{tab:misclassified_compounds_{threshold}_{affinity_type}}}",
            "    \\begin{smarttabular}{clcccc}",
            "        \\toprule",
            "        \\textbf{Compound ID} & \\textbf{Protein} & \\textbf{Similarity Ratio} & \\textbf{Purity} & \\textbf{Outlier Score} & \\textbf{Error Type} \\\\",
            "        \\midrule"
        ]
        
        # Add table rows
        for entry in all_entries:
            compound_id = entry['compound_id']
            protein = entry['protein']
            
            # Format similarity ratio
            if entry['similarity_ratio'] is not None:
                similarity_ratio = f"{entry['similarity_ratio']:.3f}"
            else:
                similarity_ratio = "N/A"
            
            # Format purity as percentage
            purity_percent = f"{entry['purity_score'] * 100:.0f}\\%"
            
            # Format outlier score
            if entry['outlier_score'] is not None:
                outlier_score = f"{entry['outlier_score']:.2f}"
            else:
                outlier_score = "N/A"
            
            error_type = entry['error_type']
            
            # Create table row
            row = f"        {compound_id} & {protein} & {similarity_ratio} & {purity_percent} & {outlier_score} & {error_type} \\\\"
            latex_lines.append(row)
        
        # Table footer
        latex_lines.extend([
            "        \\bottomrule",
            "    \\end{smarttabular}",
            "\\end{table}"
        ])
        
        return '\n'.join(latex_lines)
    
    def generate_complete_document_with_groups(self, grouped_files: Dict[Tuple[str, str], List[Path]]):
        """Generate a complete LaTeX document organized by threshold/affinity groups."""
        document_lines = [
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage{float}",
            "\\usepackage{array}",
            "",
            "% Define smarttabular environment if not available",
            "\\newenvironment{smarttabular}[1]{%",
            "    \\begin{tabular}{#1}%",
            "}{%",
            "    \\end{tabular}%",
            "}",
            "",
            "\\title{Misclassification Analysis Tables by Threshold and Affinity Type}",
            "\\author{Generated by misclassification\\_to\\_latex.py}",
            "\\date{\\today}",
            "",
            "\\begin{document}",
            "",
            "\\maketitle",
            "",
        ]
        
        # Add sections for each threshold/affinity combination
        for (threshold, affinity_type), group_files in sorted(grouped_files.items()):
            section_title = f"{threshold.replace('_', ' ').title()} - {affinity_type.upper()}"
            document_lines.extend([
                f"\\section{{{section_title}}}",
                ""
            ])
            
            # Add individual fold tables for this group
            for i, json_file in enumerate(sorted(group_files), 1):
                fold_match = re.search(r'fold_(\d+)', json_file.stem)
                fold_num = fold_match.group(1) if fold_match else str(i)
                
                latex_content = self.generate_latex_table_for_fold(json_file)
                if latex_content:
                    document_lines.extend([
                        f"\\subsection{{Fold {fold_num}}}",
                        "",
                        latex_content,
                        "",
                        "\\clearpage",
                        ""
                    ])
        
        document_lines.extend([
            "\\end{document}"
        ])
        
        # Save complete document
        complete_doc_file = self.output_dir / "complete_misclassification_report_by_groups.tex"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(complete_doc_file, 'w') as f:
            f.write('\n'.join(document_lines))
        logger.info(f"Saved complete LaTeX document to {complete_doc_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Convert misclassification analysis JSON files to LaTeX tables")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing JSON files with misclassification analysis")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save LaTeX table files")
    parser.add_argument("--pattern", type=str, default="*advanced_misclassification_analysis.json",
                       help="File pattern to match JSON files (default: *advanced_misclassification_analysis.json)")
    
    args = parser.parse_args()
    
    # Create converter and run conversion
    converter = MisclassificationToLatexConverter(args.input_dir, args.output_dir)
    converter.convert_all_files(args.pattern)
    
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
