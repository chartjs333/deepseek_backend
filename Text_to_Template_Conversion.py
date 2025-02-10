from pydoc import text
import pandas as pd
import os
import re
import logging
from typing import Set, List
from string import Formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardize_symptom(symptom: str) -> str:
    """Standardize symptom names by removing common variations and special characters"""
    if not isinstance(symptom, str):
        return ""

    symptom = symptom.lower()
    symptom = re.sub(r"(?i)_sympt$|_hp.*$", "", symptom)
    symptom = re.sub(r"[_\-/]", " ", symptom)
    symptom = re.sub(r"\s+", " ", symptom)

    replacements = {
        "parkinsonian": "parkinson",
        "psychiatric": "psychotic",
        "psychological": "psychotic",
        "dysfunction": "disorder",
        "abnormality": "disorder"
    }

    for old, new in replacements.items():
        symptom = symptom.replace(old, new)

    return symptom.strip().title()

def get_symptom_columns(df: pd.DataFrame) -> List[str]:
    """Extract symptom-related columns from DataFrame"""
    return sorted([col for col in df.columns
                   if "_sympt" in col.lower() or "_hp" in col.lower()])

def get_template_fields(template: str) -> Set[str]:
    """Extract all field names from template string using string.Formatter"""
    return {
        field_name for _, field_name, _, _
        in Formatter().parse(template)
        if field_name is not None
    }

def create_base_template() -> str:
    """Create base document template without dynamic symptoms section"""
    return """The patient with individual ID {individual_id} from family {family_id} has the following characteristics:
    - Disease: {disease_abbrev} .
    - Number of heterozygous mutations in affected individuals: {num_het_mut_affected} .
    - Number of homozygous mutations in affected individuals: {num_hom_mut_affected} .
    - Number of heterozygous mutations in unaffected individuals: {num_het_mut_unaffected} .
    - Number of homozygous mutations in unaffected individuals: {num_hom_mut_unaffected} .
    - Family history of the disease: {famhx} .
    - Number of wild-type members in affected individuals: {num_wildtype_affected} .
    - Number of wild-type members in unaffected individuals: {num_wildtype_unaffected} .
    - Genes associated with the disease: {gene1}, {gene2}, {gene3} .
    - Disease duration: {duration} years.
    - Age at diagnosis: {age_dx} .
    - Is the patient the index case: {index_pat} .
    - Age at death or "patient alive": {age_death} .
    - Additional clinical information: {clinical_info} .
    - Clinical status: {status_clinical} .
    - Patient's response to levodopa: {levodopa_response} .
    - Comments about the patient: {comments_pat} .
    - Sex of the patient: {sex} .
    - Ethnicity of the patient: {ethnicity} .
    - Age of onset: {aao} .
    - Patient's country of residence: {country} .
    - Initial symptoms: {initial_sympt1}, {initial_sympt2}, {initial_sympt3} .
    - CADD Score 1: {cadd_1}, CADD Score 2: {cadd_2}, CADD Score 3: {cadd_3} .
    - Mutation types: {mut1_type}, {mut2_type}, {mut_3_type} .
    - Genome version: {hg_version} .
    - Genome build: {genome_build} .
    - Reference alleles: {reference_allele1}, {reference_allele2}, {reference_allele3} .
    - cDNA mutations: {mut1_c}, {mut2_c}, {mut3_c} .
    - Genomic mutations: {mut1_g}, {mut2_g}, {mut3_g} .
    - Observed alleles: {observed_allele1}, {observed_allele2}, {observed_allele3} .
    - Mutation genotypes: {mut1_genotype}, {mut2_genotype}, {mut3_genotype} .
    - Pathogenicity of mutations: {pathogenicity1}, {pathogenicity2}, {pathogenicity3} .
    - Physical location of mutations: {physical_location1}, {physical_location2}, {physical_location3} .
    - Functional evidence: {fun_evidence_pos_1}, {fun_evidence_pos_2}, {fun_evidence_pos_3} .
    - Mutation aliases: {mut1_alias}, {mut2_alias}, {mut3_alias} .
    - Protein-level mutations: {mut1_p}, {mut2_p}, {mut3_p} .

    For more information, refer to the article with PMID: {pmid} .
    Study details: {study_design}, {author, year}
    Additional comments: {comments_study}
"""


def add_symptoms_to_template(symptom_columns: List[str], row: pd.Series) -> str:
    """Create dynamic symptoms section using row data"""
    symptoms_section = ""
    if symptom_columns:
        symptoms_section = "Additional symptoms and assessments\n"
        for col in symptom_columns:
            display_name = standardize_symptom(col)
            if display_name:
                col_value = str(row[col]).strip().lower()
                if col_value == "yes":
                    symptoms_section += f"\n    - {display_name}: Symptom present"
                elif col_value == "no":
                    symptoms_section += f"\n    - {display_name}: Symptom absent"

    return symptoms_section

def convert_excel_to_text(df: pd.DataFrame, output_dir: str) -> None:
    """
    Convert DataFrame data to formatted text files

    Args:
        df (pd.DataFrame): Input DataFrame with patient data
        output_dir (str): Directory to save output text files
    """
    try:
        # Convert columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        os.makedirs(output_dir, exist_ok=True)

        # Create base template
        template = create_base_template()

        # Get all required fields from template
        required_fields = get_template_fields(template)

        # Process each row
        for index, row in df.iterrows():
            try:
                patient_id = row.get("individual_id", f"unknown_{index}")
                output_file = os.path.join(output_dir, f"{index}.txt")

                # Convert row to dictionary and make all keys lowercase
                data = {k.lower(): v for k, v in row.to_dict().items()}

                # Add missing fields with "N/A" value
                for field in required_fields:
                    if field not in data:
                        data[field] = "N/A"
                    elif pd.isna(data[field]):  # Handle NaN/None values
                        data[field] = "N/A"
                    else:
                        data[field] = standardize_symptom(str(data[field])) if field in ["initial_sympt1",
                                                                                         "initial_sympt2",
                                                                                         "initial_sympt3"] else str(
                            data[field])

                symptom_columns = get_symptom_columns(df)
                result_template = template + add_symptoms_to_template(symptom_columns, row)

                text_content = result_template.format(**data)
                text_content = re.sub(r"-99", "N/A", text_content)
                # Удалить строки, где после двоеточия только N/A, пробел, точка или запятая
                text_content = re.sub(r"(?m)^.*:\s*(N/A|\.|,|\s*)+$", "", text_content)
                # Удалить N/A после двоеточия, если есть хотя бы одно слово
                text_content = re.sub(r":\s*N/A", ":", text_content)
                # Удалить строки, если после двоеточия нет слов
                text_content = re.sub(r"(?m)^.*:\s*[^a-zA-Z0-9]+$", "", text_content)
                # Удалить N/A, если есть хотя бы одно слово
                text_content = re.sub(r"(,\s*N/A)+", "", text_content)
                # Удалить строки, если после двоеточия нет слов
                text_content = re.sub(r"(?m)^.*:\s*[^a-zA-Z0-9]+$", "", text_content)
                # Удалить строки, если после двоеточия только N/A, пробел, точка или запятая
                text_content = re.sub(r"(?m)^.*:\s*(N/A|\.|,|\s*)+$", "", text_content)
                # Удалить пустые строки
                text_content = re.sub(r"(?m)^\s*$\n?", "", text_content)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text_content)

                logger.info(f"\nCreated file for patient {patient_id}")
                logger.info(text_content)

            except Exception as e:
                logger.warning(f"Error processing patient {patient_id}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise



if __name__ == "__main__":
    try:
        # Example usage
        df = pd.read_excel("D:\\000333\\deepseek_backend\\input\\TAF1_Upload_2025_02_03.xlsx")
        convert_excel_to_text(df, "output_directory")
    except Exception as e:
        logger.error(f"Failed to convert data: {str(e)}")