"""
Reusable Streamlit UI components for PicPicComparison.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
) -> None:
    """Render a styled metric card.

    Args:
        label: Metric name displayed above the value.
        value: The metric value string to display.
        delta: Optional delta string shown with arrow indicator.
        help_text: Optional tooltip text.
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text,
    )


def render_image_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    label_a: str = "Image A",
    label_b: str = "Image B",
) -> None:
    """Render two images side by side.

    Args:
        img_a: First image matrix (H x W), float64 in [0, 1].
        img_b: Second image matrix (H x W), float64 in [0, 1].
        label_a: Label for the first image.
        label_b: Label for the second image.
    """
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.subheader(label_a)
        st.image(img_a, use_container_width=True, clamp=True)
    with col_b:
        st.subheader(label_b)
        st.image(img_b, use_container_width=True, clamp=True)


def render_step_header(step_number: int, title: str) -> None:
    """Render a styled step header.

    Args:
        step_number: The step number.
        title: Step title text.
    """
    st.markdown(
        f'<div class="step-header">'
        f"<strong>Step {step_number}:</strong> {title}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_download_section(
    report_json: str,
    figures: Optional[Dict[str, plt.Figure]] = None,
) -> None:
    """Render download buttons for report and figures.

    Args:
        report_json: JSON string of the comparison report.
        figures: Optional dict mapping filenames to matplotlib Figures.
    """
    st.markdown("---")
    st.subheader("📥 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📄 Download Report (JSON)",
            data=report_json,
            file_name="comparison_report.json",
            mime="application/json",
            use_container_width=True,
        )

    if figures:
        with col2:
            for fname, fig in figures.items():
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                st.download_button(
                    label=f"🖼️ {fname}",
                    data=buf.getvalue(),
                    file_name=fname,
                    mime="image/png",
                    use_container_width=True,
                )


def render_metrics_grid(metrics: Dict[str, Any]) -> None:
    """Render a grid of metric cards.

    Args:
        metrics: Dict mapping metric names to (value_str, help_text) tuples.
    """
    cols = st.columns(min(len(metrics), 4))
    for i, (name, (value, help_text)) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            render_metric_card(label=name, value=value, help_text=help_text)


def render_kpi_strip(metrics: dict) -> None:
    """Render a horizontal KPI strip with gradient cards.

    Args:
        metrics: Dict mapping labels to (value_str, gradient_css).
    """
    cards_html = ""
    for label, (value, gradient) in metrics.items():
        cards_html += f"""
        <div class="kpi-card" style="background: {gradient};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """

    st.html(f'<div class="kpi-strip">{cards_html}</div>')
