import sys
import logging
from pathlib import Path

from src.data_loader import load_data
from src.validation import validate_and_clean_data
from src.features import engineer_features
from src.scoring import ScoringEngine
from src.explanations import generate_worklist_explanations
from src.fairness import analyze_fairness
from src.report import save_outputs, format_human_readable_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main(data_dir: str = "data", output_dir: str = "outputs"):
    """
    Main CLI entrypoint for running the complete Overpayment Signal pipeline.
    """
    logger.info("1. Loading raw dataset...")
    cases_raw, payments_raw = load_data(data_dir=data_dir)

    logger.info("2. Validating and cleaning data...")
    cases_clean, payments_clean, validation_report = validate_and_clean_data(cases_raw, payments_raw)

    logger.info("3. Engineering case-level behavioral features...")
    features_df = engineer_features(cases_clean, payments_clean)

    logger.info("4. Computing transparent risk scores and ranking...")
    engine = ScoringEngine()
    scored_full_df, subscores_df = engine.compute_scores(features_df)
    
    # Get top 20 cases
    top20_raw, top20_subscores = engine.get_ranked_worklist(scored_full_df, top_n=20)

    logger.info("5. Generating plain-language investigator explanations...")
    top20_explained = generate_worklist_explanations(top20_raw, top20_subscores)

    logger.info("6. Performing demographic fairness and disparity analysis...")
    fairness_df = analyze_fairness(scored_full_df, top20_explained)

    logger.info("7. Writing output files...")
    top20_path, fairness_path = save_outputs(top20_explained, fairness_df, output_dir=output_dir)

    logger.info("8. Displaying human-readable pipeline report...\n")
    report_text = format_human_readable_report(top20_explained, fairness_df, validation_report.summary())
    print(report_text)

    logger.info(f"Pipeline completed successfully. Outputs saved to:")
    logger.info(f"  - Top 20 Worklist: {top20_path}")
    logger.info(f"  - Fairness Report: {fairness_path}")

if __name__ == "__main__":
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    main(data_dir=data_directory, output_dir=output_directory)
