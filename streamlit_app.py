import streamlit as st
from Bio import SeqIO
import copy
import numpy as np
from scipy.optimize import curve_fit


# ------------------------------
# Gaussian-fitting functions
# ------------------------------

def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def fit_gaussian_to_peak(x_data, y_data):
    try:
        if np.all(y_data == y_data[0]):
            return {"fit_success": False}  # flat or constant, skip

        a_guess = np.max(y_data)
        mu_guess = x_data[np.argmax(y_data)]
        sigma_guess = 3.0

        popt, _ = curve_fit(gaussian, x_data, y_data, p0=[a_guess, mu_guess, sigma_guess])
        fitted = gaussian(x_data, *popt)
        residuals = y_data - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

        if ss_tot == 0 or not np.isfinite(ss_res) or not np.isfinite(ss_tot):
            return {"fit_success": False}

        r_squared = 1 - (ss_res / ss_tot)

        return {
            "a": popt[0],
            "mu": popt[1],
            "sigma": popt[2],
            "r2": r_squared,
            "fit_success": True
        }

    except Exception:
        return {"fit_success": False}

def smart_trim(trace_channel, ploc2, min_consecutive=10, window_radius=8, min_r2=0.9, sigma_range=(1.5, 15), min_height=100):
    n = len(ploc2)
    quality_flags = []
    for i in range(n):
        peak_center = ploc2[i]
        start = max(0, peak_center - window_radius)
        end = min(len(trace_channel), peak_center + window_radius + 1)
        x = np.arange(start, end)
        y = trace_channel[start:end]
        fit = fit_gaussian_to_peak(x, y)
        if fit["fit_success"]:
            is_good = (
                fit["r2"] >= min_r2 and
                sigma_range[0] <= fit["sigma"] <= sigma_range[1] and
                fit["a"] >= min_height
            )
            quality_flags.append(is_good)
        else:
            quality_flags.append(False)

    def find_good_region(flags, min_consecutive):
        for i in range(len(flags) - min_consecutive + 1):
            if all(flags[i:i + min_consecutive]):
                start = i
                break
        else:
            return None, None
        for j in range(len(flags) - 1, start + min_consecutive - 1, -1):
            if all(flags[j - min_consecutive + 1:j + 1]):
                end = j
                break
        else:
            return None, None
        return start, end + 1

    return find_good_region(quality_flags, min_consecutive)


# ------------------------------
# Trimming wrapper
# ------------------------------

def run_trimming_strategy(record_raw, method="abi-trim", channel_key="A"):
    raw = record_raw.annotations["abif_raw"]
    basecalls = str(record_raw.seq)

    if method == "abi-trim":
        trimmed_copy = copy.deepcopy(record_raw)
        record_trimmed = SeqIO.read(trimmed_copy, "abi-trim")
        trimmed_seq = str(record_trimmed.seq)
        for i in range(len(basecalls) - len(trimmed_seq) + 1):
            if basecalls[i:i + len(trimmed_seq)] == trimmed_seq:
                return i, i + len(trimmed_seq)

    elif method == "gaussian":
        ploc2 = raw["PLOC2"]
        channels = {
            "G": np.array(raw.get("DATA9", []), dtype=float),
            "A": np.array(raw.get("DATA10", []), dtype=float),
            "T": np.array(raw.get("DATA11", []), dtype=float),
            "C": np.array(raw.get("DATA12", []), dtype=float),
        }
        trim_start, trim_end = smart_trim(channels[channel_key], ploc2)
        return trim_start, trim_end

    else:
        raise ValueError("Unknown trimming method")


# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="Het Detector", layout="wide")
st.title("🔬 Sanger Heterozygosity Detector")
st.caption("Uploads `.ab1` files, trims with Mott or Gaussian fitting, and flags potential heterozygous sites.")

# Sidebar controls
trimming_method = st.selectbox("Trimming Method", ["abi-trim", "gaussian"])
het_threshold = st.slider("Heterozygosity Threshold (2nd / top peak ratio)", 0.1, 1.0, 0.33, 0.01)
min_phred = st.slider("Minimum PHRED score to include", 0, 60, 20)

# File upload
uploaded_files = st.file_uploader("Upload `.ab1` files", type=["ab1"], accept_multiple_files=True)

if uploaded_files:
    het_rows = []

    for file in uploaded_files:
        try:
            record_raw = SeqIO.read(file, "abi")
        except Exception as e:
            st.error(f"Failed to read {file.name}: {e}")
            continue

        try:
            trim_start, trim_end = run_trimming_strategy(record_raw, method=trimming_method, channel_key="A")
            if trim_start is None or trim_end is None:
                st.warning(f"No valid trim region found in {file.name} using method '{trimming_method}'. Skipping.")
                continue
        except Exception as e:
            st.warning(f"Trimming failed for {file.name}: {e}")
            continue


        raw = record_raw.annotations["abif_raw"]
        ploc2 = raw["PLOC2"]
        called_bases = raw["PBAS2"].decode()
        phred_scores = record_raw.letter_annotations["phred_quality"]

        channels = {
            "G": np.array(raw.get("DATA9", []), dtype=float),
            "A": np.array(raw.get("DATA10", []), dtype=float),
            "T": np.array(raw.get("DATA11", []), dtype=float),
            "C": np.array(raw.get("DATA12", []), dtype=float),
        }

        for i in range(trim_start, trim_end):
            if phred_scores[i] < min_phred:
                continue

            base = called_bases[i]
            trace_idx = ploc2[i]
            intensities = {nt: channels[nt][trace_idx] for nt in "ATCG"}
            sorted_intensity = sorted(intensities.items(), key=lambda x: x[1], reverse=True)
            top_nt, top_val = sorted_intensity[0]
            second_nt, second_val = sorted_intensity[1]
            ratio = second_val / top_val if top_val > 0 else 0

            if ratio >= het_threshold:
                het_rows.append({
                    "File": file.name,
                    "Base Index": i+1,
                    "Base Call": base,
                    "Primary": f"{top_nt} ({top_val:.1f})",
                    "Secondary": f"{second_nt} ({second_val:.1f})",
                    "Ratio": round(ratio, 2),
                    "PHRED": phred_scores[i],
                })

    if het_rows:
        st.subheader("⚠️ Potential Heterozygous Sites")
        st.dataframe(het_rows, use_container_width=True)
    else:
        st.success("No potential heterozygous sites detected.")
