"""ui/pages/export.py — Phase 7: Export page."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st

from db.session_store import get_messages
from utils.serialization import b64_to_obj
from ui.components import alert, badge, metrics_row
from ui.graph_helpers import get_state


def page_export(_rt):
    s = st.session_state.pipeline_state
    if not s.get("eval_metrics") and not s.get("best_model_key"):
        alert("Complete the Evaluation phase first.", "warning")
        return

    tid       = st.session_state.tid
    metrics   = s.get("eval_metrics", {})
    prob_type = s.get("problem_type", "classification")
    best_model= s.get("best_model_key", "—")
    cv_score  = s.get("best_cv_score")
    baseline  = s.get("_baseline_score")

    # Register this run in the comparison store so Compare Datasets page is populated
    from ui.state_store import add_dataset
    dataset_name = getattr(st.session_state, "dataset_name", None) or "dataset.csv"
    add_dataset(dataset_name, tid, metrics, pipeline_state=s)

    st.markdown(f'{badge("Phase 7")} <h1 style="display:inline;margin-left:.5rem;">Export</h1>', unsafe_allow_html=True)
    st.caption("Download reports, scripts, model artifacts, and run predictions.")

    items = [
        ("Target",       s.get("target", "—")),
        ("Problem type", prob_type),
        ("Best model",   best_model),
        ("CV score",     f"{cv_score:.4f}" if cv_score else "—"),
    ]
    if baseline is not None:
        items.append(("Naive baseline", f"{baseline:.4f}"))
    if cv_score and baseline is not None:
        items.append(("Lift vs baseline", f"+{cv_score - baseline:.4f}"))
    metrics_row(items)

    if metrics:
        st.markdown("### Evaluation metrics")
        metric_map = {}
        if prob_type == "classification":
            for k in ["f1_weighted", "accuracy", "roc_auc", "precision_weighted", "recall_weighted"]:
                if metrics.get(k) is not None:
                    metric_map[k] = round(metrics[k], 4)
        else:
            for k in ["r2", "rmse", "mae", "mse"]:
                if metrics.get(k) is not None:
                    metric_map[k] = round(metrics[k], 4)
        if metric_map:
            st.dataframe(pd.DataFrame([metric_map]), width="stretch")

    shap_top = (s.get("shap_importance") or [])[:5]
    if shap_top:
        st.markdown("### Top 5 SHAP features")
        shap_df = pd.DataFrame([{"Feature": f["feature"], "Importance": round(f["importance"], 4)} for f in shap_top])
        st.dataframe(shap_df, width="stretch")

    st.markdown("---")
    st.markdown("### 📥 Downloads")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF…"):
                try:
                    full_state = get_state(tid)
                    if full_state.get("_preprocessor_b64") and not _rt.get_key(tid, "preprocessor"):
                        _rt.set_key(tid, "preprocessor", b64_to_obj(full_state["_preprocessor_b64"]))
                    if full_state.get("_label_encoder_b64") and not _rt.get_key(tid, "label_encoder"):
                        _rt.set_key(tid, "label_encoder", b64_to_obj(full_state["_label_encoder_b64"]))
                    from report_generator import generate_report
                    msgs = get_messages(tid)
                    pdf_bytes = generate_report(full_state, st.session_state.dataset_name or "dataset.csv", msgs or None)
                    fname = (st.session_state.dataset_name or "dataset").replace(".csv", "").replace(" ", "_") + "_automl_report.pdf"
                    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=fname, mime="application/pdf")
                except ModuleNotFoundError as e:
                    if e.name == "reportlab":
                        alert(
                            "PDF export requires the `reportlab` package. Install dependencies again after updating `requirements.txt`, then retry.",
                            "error",
                        )
                    else:
                        alert(f"PDF generation failed: {e}", "error")
                except Exception as e:
                    alert(f"PDF generation failed: {e}", "error")

    with col_b:
        if st.button("🐍 Generate Python Script"):
            with st.spinner("Generating script…"):
                try:
                    full_state = get_state(tid)
                    from script_exporter import generate_python_script
                    script_code = generate_python_script(full_state, st.session_state.dataset_name or "dataset.csv")

                    # Fix #11: validate generated script with ast.parse before offering download
                    import ast as _ast
                    try:
                        _ast.parse(script_code)
                    except SyntaxError as syn_err:
                        alert(f"⚠️ Generated script has a syntax error — please report this bug: {syn_err}", "warning")
                        # Still allow download so user can inspect/fix
                    fname = (st.session_state.dataset_name or "dataset").replace(".csv", "").replace(" ", "_") + "_automl_pipeline.py"
                    st.download_button("⬇️ Download Python Script", data=script_code.encode(), file_name=fname, mime="text/x-python")
                except Exception as e:
                    alert(f"Script generation failed: {e}", "error")

    with col_c:
        if st.button("🤖 Export Model (.joblib)"):
            with st.spinner("Exporting model…"):
                try:
                    preprocessor   = _rt.get_key(tid, "preprocessor")
                    best_model_obj = _rt.get_key(tid, "best_model")
                    if preprocessor is None or best_model_obj is None:
                        full_state = get_state(tid)
                        if preprocessor is None and full_state.get("_preprocessor_b64"):
                            preprocessor = b64_to_obj(full_state["_preprocessor_b64"])
                        if best_model_obj is None and full_state.get("_best_model_b64"):
                            best_model_obj = b64_to_obj(full_state["_best_model_b64"])
                        if best_model_obj is None:
                            from utils.runtime import get_or_restore_runtime
                            rt_r, _ = get_or_restore_runtime(full_state, required=("best_model",))
                            best_model_obj = rt_r.get("best_model")
                        if preprocessor is None:
                            from utils.runtime import get_or_restore_runtime
                            rt_r, _ = get_or_restore_runtime(full_state, required=("preprocessor",))
                            preprocessor = rt_r.get("preprocessor")
                    if preprocessor is None or best_model_obj is None:
                        alert("Model not in memory — re-run the pipeline.", "error")
                    else:
                        from utils.ml_helpers import build_export_pipeline
                        buf = build_export_pipeline(preprocessor, best_model_obj)
                        st.download_button("⬇️ Download Model", data=buf,
                                           file_name="automl_full_pipeline.joblib",
                                           mime="application/octet-stream")
                except Exception as e:
                    alert(f"Model export failed: {e}", "error")

    st.markdown("---")
    st.markdown("### 🔮 Predict on new data")
    pred_file = st.file_uploader("Upload CSV for predictions", type=["csv"], key="pred_upload")
    if pred_file:
        if st.button("⚡ Run predictions"):
            with st.spinner("Running predictions…"):
                try:
                    preprocessor   = _rt.get_key(tid, "preprocessor")
                    best_model_obj = _rt.get_key(tid, "best_model")
                    label_encoder  = _rt.get_key(tid, "label_encoder")
                    if preprocessor is None or best_model_obj is None:
                        full_state = get_state(tid)
                        if full_state.get("_preprocessor_b64"):
                            preprocessor = b64_to_obj(full_state["_preprocessor_b64"])
                        if full_state.get("_best_model_b64"):
                            best_model_obj = b64_to_obj(full_state["_best_model_b64"])
                        if full_state.get("_label_encoder_b64") and label_encoder is None:
                            label_encoder = b64_to_obj(full_state["_label_encoder_b64"])
                        if best_model_obj is None:
                            from utils.runtime import get_or_restore_runtime
                            rt_r, _ = get_or_restore_runtime(full_state, required=("best_model",))
                            best_model_obj = rt_r.get("best_model")
                    if preprocessor is None or best_model_obj is None:
                        alert("Pipeline not in memory — re-run the pipeline.", "error")
                    else:
                        content = pred_file.read()
                        new_df  = pd.read_csv(io.BytesIO(content))
                        nc      = new_df.select_dtypes(include="number").columns
                        new_df[nc] = new_df[nc].replace([np.inf, -np.inf], np.nan)
                        preds = best_model_obj.predict(preprocessor.transform(new_df))
                        if label_encoder is not None:
                            try:
                                preds = label_encoder.inverse_transform(preds)
                            except Exception:
                                pass
                        result_df = new_df.copy()
                        result_df["prediction"] = preds
                        st.dataframe(result_df.head(10), width="stretch")
                        csv_out = result_df.to_csv(index=False).encode()
                        st.download_button("⬇️ Download predictions.csv", data=csv_out,
                                           file_name="predictions.csv", mime="text/csv")
                except Exception as e:
                    alert(f"Prediction failed: {e}", "error")

    with st.expander("💬 Agent message log"):
        msgs = get_messages(tid) or []
        if msgs:
            for m in msgs[-50:]:
                role    = m.get("role", "")
                content = m.get("content", "")
                st.markdown(f"**{role}**: {content}")
        else:
            agent_msgs = s.get("agent_messages", [])
            for m in agent_msgs[-50:]:
                if isinstance(m, dict):
                    st.markdown(f"**{m.get('type','')}**: {m.get('content','')}")
                else:
                    st.markdown(str(m))
