import streamlit as st
from pathlib import Path
import joblib
import pandas as pd

st.set_page_config(
    page_title="Payment Review",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_model():
    path = Path(__file__).parent / "data" / "fraud_model.joblib"
    return joblib.load(path)


def score_csv(data, currency, rate, model_bundle):
    """Validate required inputs and score without changing original amounts."""
    required = model_bundle["features"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError("The CSV contains no transaction rows.")
    if len(data) > 250000:
        raise ValueError("Upload up to 250,000 transactions per file.")
    reserved = {"input_row", "amount_usd", "input_currency", "ngn_per_usd",
                "fraud_score", "flag_for_review", "outside_amount_range"}
    if reserved.intersection(data.columns):
        raise ValueError("Upload the original transaction CSV, not scored results.")
    X = data[required].apply(pd.to_numeric, errors="coerce")
    invalid = X.isna() | X.isin([float("inf"), float("-inf")])
    if invalid.any().any():
        columns = invalid.columns[invalid.any()].tolist()
        raise ValueError("Missing or nonnumeric values in: " + ", ".join(columns))
    if (X["amount"] <= 0).any() or (X["distance"] < 0).any():
        raise ValueError("Amounts must be positive and distances nonnegative.")
    for column in ["credit_card", "remote"]:
        if not X[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1.")
    if not (X["hour"].between(0, 23) & X["hour"].mod(1).eq(0)).all():
        raise ValueError("Hour must be a whole number from 0 to 23.")
    if currency not in ["USD", "NGN"]:
        raise ValueError("Choose USD or NGN for the amount column.")
    if currency == "NGN":
        if rate is None or not 0 < rate < float("inf"):
            raise ValueError("Enter a positive exchange rate (naira per US dollar).")
        X["amount"] = X["amount"] / rate
    results = data.copy()
    results.insert(0, "input_row", range(1, len(data) + 1))
    results["input_currency"] = currency
    results["ngn_per_usd"] = rate if currency == "NGN" else None
    results["amount_usd"] = X["amount"]
    results["outside_amount_range"] = ~X["amount"].between(0.206971, 744.501355)
    results["fraud_score"] = model_bundle["model"].predict_proba(X)[:, 1]
    results["flag_for_review"] = results["fraud_score"] >= model_bundle["threshold"]
    return results


bundle = load_model()

with st.sidebar:
    st.title("🛡️ Payment Review")
    st.caption("Fraud detection workspace")

    page = st.radio(
        "Navigate",
        ["Check a transaction", "Review a CSV"]
    )

    st.divider()
    st.caption("Built with synthetic CardSim data.")


    with st.expander("About the model"):
        st.write("Class-weighted gradient-boosted trees")
        st.caption(f"Fixed review threshold: {bundle['threshold']:.2f}")
        st.markdown(
            "**Final test results**\n\n"
            "- Fraud recall: **94.72%**\n"
            "- Alert precision: **37.20%**\n"
            "- Accuracy: **98.28%**\n"
            "- Average precision: **0.7686**"
        )
        st.caption(
            "Later test period: May 25–June 29, 2023. "
            "305 fraud cases caught, 17 missed; 515 false alerts. "
            "These are recorded test results, not live metrics for an upload."
        )
        st.caption(
            "Trained on synthetic CardSim data based on US payment surveys. "
            "Naira inputs are converted using your entered rate; "
            "the model has not been validated on Nigerian payments. "
            "Scores are not calibrated probabilities and flags do not confirm fraud."
        )
        st.caption(
            "Limitation: full-dataset exploration preceded splitting, "
            "so the test period is not fully independent of exploration."
        )

st.title(page)
st.caption("Explore payment alerts. Make informed review decisions.")

if page == "Check a transaction":
    form_column, result_column = st.columns([3, 2], gap="large")

    with form_column:
        with st.form("transaction_form"):
            st.subheader("Transaction details")

            amount_ngn = st.number_input(
                "Amount (₦)",
                min_value=0.01,
                value=25000.0,
                step=1000.0
            )

            naira_per_dollar = st.number_input(
                "Exchange rate (₦ per US$)",
                min_value=0.01,
                value=None,
                step=1.0,
                placeholder="Enter your chosen conversion rate",
                help="Naira required for one US dollar. Enter a positive rate; "
                     "this app does not fetch live exchange rates."
            )
            st.caption(
                "Naira input is converted to US dollars for the synthetic model. "
                "This model has not been validated on Nigerian payments."
            )

            distance = st.number_input(
                "Distance (grid units)",
                min_value=0.0,
                value=85.0
            )

            card_type = st.selectbox(
                "Card type",
                ["Debit", "Credit"]
            )

            location = st.selectbox(
                "Payment location",
                ["In-person", "Remote"]
            )

            hour = st.slider("Hour of day", 0, 23, 12)

            submitted = st.form_submit_button(
                "Check transaction",
                type="primary",
                width="stretch"
            )

    with result_column:
        with st.container(border=True):
            st.subheader("Review result")

            if submitted and naira_per_dollar is None:
                st.info("Enter an exchange rate to check this transaction.")

            elif submitted:
                amount_usd = amount_ngn / naira_per_dollar
                st.caption(
                    f"₦{amount_ngn:,.2f} ÷ {naira_per_dollar:,.2f} = "
                    f"US${amount_usd:,.2f} used by the model."
                )
                if not 0.206971 <= amount_usd <= 744.501355:
                    st.warning(
                        "Converted amount is outside the dataset's observed "
                        "range (about US$0.21–744.50). This score is outside "
                        "the amount range evaluated in the project."
                    )
                transaction = pd.DataFrame([{
                    "amount": amount_usd,
                    "distance": distance,
                    "credit_card": int(card_type == "Credit"),
                    "remote": int(location == "Remote"),
                    "hour": hour
                }])

                score = float(
                    bundle["model"].predict_proba(
                        transaction[bundle["features"]]
                    )[0, 1]
                )

                st.metric("Model score", f"{score:.3f}")
                st.caption(
                    f"Review threshold: {bundle['threshold']:.2f}"
                )

                if score >= bundle["threshold"]:
                    st.warning("Review recommended")
                else:
                    st.success("Below review threshold")

                st.caption(
                    "The score is not a calibrated fraud probability."
                )

            else:
                st.info(
                    "Enter the details, then check the transaction."
                )

            st.caption(
                "The model recommends review; it does not confirm fraud."
            )

else:
    st.subheader("Review a batch of payments")
    st.caption(
        "Upload up to 250,000 rows. Original amounts are preserved; "
        "only the model input is converted when you select naira."
    )

    with st.expander("CSV format and example"):
        st.write(
            "Required columns: amount, distance, credit_card, remote, hour. "
            "Use 0 for debit / in-person and 1 for credit / remote. "
            "Hour must be 0–23; distance uses simulator grid units. "
            "Extra columns, including known fraud labels, are preserved "
            "but are not supplied to the model."
        )
        st.download_button(
            "Download example CSV (USD)",
            "amount,distance,credit_card,remote,hour\n25,85,0,0,12\n100,150,1,1,1\n",
            file_name="example_transactions_usd.csv",
            mime="text/csv"
        )

    uploaded = st.file_uploader("Transaction CSV", type=["csv"])
    currency_label = st.radio(
        "Currency of the amount column",
        ["US dollars (USD)", "Naira (NGN)"],
        horizontal=True,
        help="Select USD for the original CardSim dataset."
    )
    currency = "NGN" if currency_label == "Naira (NGN)" else "USD"
    rate = None
    if currency == "NGN":
        rate = st.number_input(
            "Exchange rate (₦ per US$)", min_value=0.01,
            value=None, step=1.0, key="csv_rate",
            placeholder="Enter your chosen conversion rate"
        )
        st.caption("No live rate is fetched. This is a synthetic US-based model.")
    else:
        st.caption("Amounts will be passed to the model without currency conversion.")

    if uploaded is not None:
        try:
            # Bound parsing before the model sees the upload.
            source_data = pd.read_csv(uploaded, nrows=250001)
            with st.spinner("Checking and scoring transactions…"):
                results = score_csv(source_data, currency, rate, bundle)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(str(error))
        else:
            total = len(results)
            flagged = int(results["flag_for_review"].sum())
            with st.container(horizontal=True):
                st.metric("Transactions", f"{total:,}", border=True)
                st.metric("Review recommended", f"{flagged:,}", border=True)
                st.metric("Flagged share", f"{flagged / total:.1%}", border=True)

            outside = int(results["outside_amount_range"].sum())
            if outside:
                st.warning(
                    f"{outside:,} converted amounts fall outside the dataset's "
                    "observed range (about US$0.21–744.50). "
                    "Scores for these rows have not been evaluated for this range."
                )
            st.caption(
                f"Fixed review threshold: {bundle['threshold']:.2f}. "
                "Flags are review recommendations, not confirmed fraud. "
                "Scores are not calibrated probabilities."
            )
            flagged_only = st.toggle("Show flagged transactions only")
            view = results[results["flag_for_review"]] if flagged_only else results
            view = view.sort_values("fraud_score", ascending=False, kind="stable")
            priority_columns = [
                "flag_for_review", "fraud_score", "input_row", "amount",
                "input_currency", "amount_usd", "outside_amount_range",
                "distance", "credit_card", "remote", "hour"
            ]
            display_columns = priority_columns + [
                column for column in view.columns
                if column not in priority_columns
            ]
            st.dataframe(
                view[display_columns], hide_index=True, width="stretch",
                column_config={
                    "fraud_score": st.column_config.ProgressColumn(
                        "Model score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "flag_for_review": st.column_config.CheckboxColumn("Review flag"),
                    "amount_usd": st.column_config.NumberColumn(
                        "Model amount (US$)", format="%.2f"
                    )
                },
                alt="Transactions ranked by model score, highest first"
            )
            st.download_button(
                "Download all scored transactions",
                results.to_csv(index=False).encode("utf-8-sig"),
                file_name="reviewed_transactions.csv", mime="text/csv",
                type="primary"
            )
    else:
        st.info("Upload a CSV to see review flags and download scored results.")
