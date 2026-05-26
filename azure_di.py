# extractor/azure_di.py
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    AnalyzeOutputOption,
    DocumentAnalysisFeature,
)

def get_client() -> DocumentIntelligenceClient:
    endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
    key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]
    return DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

def extract_layout(file_path: str) -> AnalyzeResult:
    """
    Core extraction: text, tables, paragraphs (with roles), figures.
    Uses prebuilt-layout with key-value pairs feature enabled.
    """
    client = get_client()
    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            features=[DocumentAnalysisFeature.KEY_VALUE_PAIRS],
        )
    return poller.result()

def extract_with_figures(file_path: str) -> tuple[AnalyzeResult, dict]:
    """
    For v1: also extracts diagram/figure crops as PNG bytes.
    Returns (result, {figure_id: png_bytes})
    """
    client = get_client()
    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            output=[AnalyzeOutputOption.FIGURES],
        )
    result: AnalyzeResult = poller.result()
    operation_id = poller.details["operation_id"]

    figures_data = {}
    if result.figures:
        for figure in result.figures:
            if figure.id:
                response = client.get_analyze_result_figure(
                    model_id=result.model_id,
                    result_id=operation_id,
                    figure_id=figure.id
                )
                figures_data[figure.id] = b"".join(response)

    return result, figures_data