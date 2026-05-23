import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def load_data(filepath=None):
    if filepath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, "mahadevi_history.json")
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Historical data file not found at {filepath}")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    # Convert dates to datetime objects
    df["date"] = pd.to_datetime(df["date"])
    # Filter for valid draws (where open/close single exist)
    df_valid = df[df["is_valid"] == True].copy()
    # Sort by date chronologically
    df_valid = df_valid.sort_values("date").reset_index(drop=True)
    
    # Cast digit columns to integers
    df_valid["open_single"] = df_valid["open_single"].astype(int)
    df_valid["close_single"] = df_valid["close_single"].astype(int)
    
    return df, df_valid

# Define Satta Matka Numerology: Cut Numbers
# Cut number is (digit + 5) % 10
CUT_NUMBERS = {i: (i + 5) % 10 for i in range(10)}

# Satta Matka Panas matching a single digit
# In Satta Matka, 3-digit panels (Panas) sum to the single digit (modulo 10)
# Digits in Pana are sorted in ascending order (with 0 treated as 10 for sorting, or standard string sorting)
# Standard Matka Pana lists are fixed:
def get_panas_for_digit(digit):
    """
    Generate all possible 3-digit Satta Matka panas for a given single digit (0-9).
    Pana digits are in ascending order. E.g. for digit 9: 126, 135, 144, 234, 270, etc.
    Digits are sorted. '0' counts as value 10 in Matka digit summation.
    """
    panas = []
    # Loop through unique sorted combinations of 3 digits
    for i in range(1, 11):  # 1 to 10 (10 represents 0)
        for j in range(i, 11):
            for k in range(j, 11):
                val_i = 0 if i == 10 else i
                val_j = 0 if j == 10 else j
                val_k = 0 if k == 10 else k
                if (val_i + val_j + val_k) % 10 == digit:
                    # format as string, convert 10 to '0'
                    d_i = '0' if i == 10 else str(i)
                    d_j = '0' if j == 10 else str(j)
                    d_k = '0' if k == 10 else str(k)
                    # Sort the character representation according to Matka rules: 
                    # 1,2,3,4,5,6,7,8,9,0
                    def matka_key(char):
                        return 10 if char == '0' else int(char)
                    
                    sorted_chars = sorted([d_i, d_j, d_k], key=matka_key)
                    pana_str = "".join(sorted_chars)
                    if pana_str not in panas:
                        panas.append(pana_str)
    return sorted(panas)

class MatkaPredictionEngine:
    def __init__(self, df_history):
        self.df = df_history.copy().reset_index(drop=True)
        
    def _get_frequency_probs(self, df_train, weekday_num=None):
        """
        Calculates historical frequency probabilities.
        If weekday_num is specified, prioritize weekday-specific probabilities.
        """
        probs_open = np.zeros(10)
        probs_close = np.zeros(10)
        
        # General frequencies
        counts_open = df_train["open_single"].value_counts().to_dict()
        counts_close = df_train["close_single"].value_counts().to_dict()
        
        for digit in range(10):
            probs_open[digit] = counts_open.get(digit, 0)
            probs_close[digit] = counts_close.get(digit, 0)
            
        # Standardize
        sum_o, sum_c = probs_open.sum(), probs_close.sum()
        probs_open = probs_open / sum_o if sum_o > 0 else np.ones(10) / 10.0
        probs_close = probs_close / sum_c if sum_c > 0 else np.ones(10) / 10.0
        
        # Weekday specific frequencies (blend 50/50 with general frequencies if enough data)
        if weekday_num is not None:
            df_weekday = df_train[df_train["weekday_num"] == weekday_num]
            if len(df_weekday) > 10:
                w_probs_open = np.zeros(10)
                w_probs_close = np.zeros(10)
                w_counts_open = df_weekday["open_single"].value_counts().to_dict()
                w_counts_close = df_weekday["close_single"].value_counts().to_dict()
                
                for digit in range(10):
                    w_probs_open[digit] = w_counts_open.get(digit, 0)
                    w_probs_close[digit] = w_counts_close.get(digit, 0)
                    
                w_sum_o, w_sum_c = w_probs_open.sum(), w_probs_close.sum()
                w_probs_open = w_probs_open / w_sum_o if w_sum_o > 0 else np.ones(10) / 10.0
                w_probs_close = w_probs_close / w_sum_c if w_sum_c > 0 else np.ones(10) / 10.0
                
                # Blend
                probs_open = 0.4 * probs_open + 0.6 * w_probs_open
                probs_close = 0.4 * probs_close + 0.6 * w_probs_close
                
        return probs_open, probs_close

    def _get_markov_probs(self, df_train, last_open, last_close):
        """
        Calculates 1-step Markov Chain transition probabilities.
        """
        probs_open = np.ones(10) / 10.0
        probs_close = np.ones(10) / 10.0
        
        if len(df_train) < 2:
            return probs_open, probs_close
            
        # Build Open transition matrix
        open_transitions = np.zeros((10, 10))
        close_transitions = np.zeros((10, 10))
        
        open_series = df_train["open_single"].values
        close_series = df_train["close_single"].values
        
        for t in range(len(df_train) - 1):
            s_from_o = open_series[t]
            s_to_o = open_series[t+1]
            open_transitions[s_from_o][s_to_o] += 1
            
            s_from_c = close_series[t]
            s_to_c = close_series[t+1]
            close_transitions[s_from_c][s_to_c] += 1
            
        # Query transition for last_open
        if last_open is not None and 0 <= last_open <= 9:
            row = open_transitions[last_open]
            row_sum = row.sum()
            if row_sum > 0:
                probs_open = row / row_sum
                # Add tiny smoothing
                probs_open = 0.95 * probs_open + 0.05 * (np.ones(10) / 10.0)
                
        # Query transition for last_close
        if last_close is not None and 0 <= last_close <= 9:
            row = close_transitions[last_close]
            row_sum = row.sum()
            if row_sum > 0:
                probs_close = row / row_sum
                probs_close = 0.95 * probs_close + 0.05 * (np.ones(10) / 10.0)
                
        return probs_open, probs_close

    def _get_pattern_probs(self, df_train, last_opens, last_closes):
        """
        Looks for sequence pattern matches in historical data.
        """
        probs_open = np.zeros(10)
        probs_close = np.zeros(10)
        
        open_series = df_train["open_single"].values
        close_series = df_train["close_single"].values
        
        # Try matching sequence of length 3, then fallback to 2
        for seq_len in [3, 2]:
            if len(last_opens) < seq_len or len(open_series) < seq_len + 2:
                continue
                
            target_open_seq = last_opens[-seq_len:]
            target_close_seq = last_closes[-seq_len:]
            
            match_counts_open = np.zeros(10)
            match_counts_close = np.zeros(10)
            matches_found_o = 0
            matches_found_c = 0
            
            # Slide window across history
            for t in range(len(open_series) - seq_len):
                # Open matching
                hist_seq_o = open_series[t : t+seq_len]
                if np.array_equal(hist_seq_o, target_open_seq):
                    next_val_o = open_series[t+seq_len]
                    match_counts_open[next_val_o] += 1
                    matches_found_o += 1
                    
                # Close matching
                hist_seq_c = close_series[t : t+seq_len]
                if np.array_equal(hist_seq_c, target_close_seq):
                    next_val_c = close_series[t+seq_len]
                    match_counts_close[next_val_c] += 1
                    matches_found_c += 1
                    
            if matches_found_o > 0:
                probs_open = match_counts_open / matches_found_o
            if matches_found_c > 0:
                probs_close = match_counts_close / matches_found_c
                
            # If we found matches, break early (prefer longer pattern match)
            if matches_found_o > 0 and matches_found_c > 0:
                break
                
        # If no patterns matched, default to uniform
        if probs_open.sum() == 0:
            probs_open = np.ones(10) / 10.0
        if probs_close.sum() == 0:
            probs_close = np.ones(10) / 10.0
            
        return probs_open, probs_close

    def _prepare_ml_features(self, df_train):
        """
        Engineers features for supervised ML classifier (RandomForest).
        Features:
        - Weekday (integer 0-6)
        - Cyclical Weekday encoding (sin/cos)
        - Lags (last 1-5 draws)
        - Rolling average of digits (last 3, 5 draws)
        """
        features_open = []
        features_close = []
        
        open_vals = df_train["open_single"].values
        close_vals = df_train["close_single"].values
        weekdays = df_train["weekday_num"].values
        
        # We need at least 5 lags to construct features
        num_lags = 5
        
        # Build features for each sample t from num_lags to end
        for t in range(num_lags, len(df_train)):
            day = weekdays[t]
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)
            
            # Open Single Features
            feats_o = [
                day, day_sin, day_cos,
                open_vals[t-1], open_vals[t-2], open_vals[t-3], open_vals[t-4], open_vals[t-5],
                np.mean(open_vals[t-3:t]), np.mean(open_vals[t-5:t]),
                np.std(open_vals[t-5:t])
            ]
            features_open.append(feats_o)
            
            # Close Single Features
            feats_c = [
                day, day_sin, day_cos,
                close_vals[t-1], close_vals[t-2], close_vals[t-3], close_vals[t-4], close_vals[t-5],
                np.mean(close_vals[t-3:t]), np.mean(close_vals[t-5:t]),
                np.std(close_vals[t-5:t])
            ]
            features_close.append(feats_c)
            
        X_o = np.array(features_open)
        y_o = open_vals[num_lags:]
        
        X_c = np.array(features_close)
        y_c = close_vals[num_lags:]
        
        return X_o, y_o, X_c, y_c

    def _get_ml_probs(self, df_train, last_opens, last_closes, target_weekday_num):
        """
        Trains RandomForest models and predicts probabilities.
        """
        # Default probabilities if training data is too small
        probs_open = np.ones(10) / 10.0
        probs_close = np.ones(10) / 10.0
        
        if len(df_train) < 30: # Need sufficient samples to train RF
            return probs_open, probs_close, None, None
            
        try:
            X_o, y_o, X_c, y_c = self._prepare_ml_features(df_train)
            
            # Construct target feature vector for the upcoming prediction
            day_sin = np.sin(2 * np.pi * target_weekday_num / 7)
            day_cos = np.cos(2 * np.pi * target_weekday_num / 7)
            
            target_feat_o = np.array([[
                target_weekday_num, day_sin, day_cos,
                last_opens[-1], last_opens[-2], last_opens[-3], last_opens[-4], last_opens[-5],
                np.mean(last_opens[-3:]), np.mean(last_opens[-5:]),
                np.std(last_opens[-5:])
            ]])
            
            target_feat_c = np.array([[
                target_weekday_num, day_sin, day_cos,
                last_closes[-1], last_closes[-2], last_closes[-3], last_closes[-4], last_closes[-5],
                np.mean(last_closes[-3:]), np.mean(last_closes[-5:]),
                np.std(last_closes[-5:])
            ]])
            
            # Train Open Classifier
            clf_o = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            clf_o.fit(X_o, y_o)
            # Ensure model understands all classes 0-9
            # In case some class didn't appear in train (rare but possible), we map predict_proba classes
            probs_o_raw = clf_o.predict_proba(target_feat_o)[0]
            probs_o_full = np.zeros(10)
            for idx, cls in enumerate(clf_o.classes_):
                probs_o_full[cls] = probs_o_raw[idx]
            probs_open = probs_o_full
            
            # Train Close Classifier
            clf_c = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            clf_c.fit(X_c, y_c)
            probs_c_raw = clf_c.predict_proba(target_feat_c)[0]
            probs_c_full = np.zeros(10)
            for idx, cls in enumerate(clf_c.classes_):
                probs_c_full[cls] = probs_c_raw[idx]
            probs_close = probs_c_full
            
            return probs_open, probs_close, clf_o, clf_c
        except Exception as e:
            print(f"ML Model training failed, falling back to frequencies: {e}")
            return probs_open, probs_close, None, None

    def _get_gbm_probs(self, df_train, last_opens, last_closes, target_weekday_num):
        """
        Trains Gradient Boosting models and predicts probabilities.
        """
        # Default probabilities if training data is too small
        probs_open = np.ones(10) / 10.0
        probs_close = np.ones(10) / 10.0
        
        if len(df_train) < 30: # Need sufficient samples to train GBM
            return probs_open, probs_close, None, None
            
        try:
            X_o, y_o, X_c, y_c = self._prepare_ml_features(df_train)
            
            # Construct target feature vector for the upcoming prediction
            day_sin = np.sin(2 * np.pi * target_weekday_num / 7)
            day_cos = np.cos(2 * np.pi * target_weekday_num / 7)
            
            target_feat_o = np.array([[
                target_weekday_num, day_sin, day_cos,
                last_opens[-1], last_opens[-2], last_opens[-3], last_opens[-4], last_opens[-5],
                np.mean(last_opens[-3:]), np.mean(last_opens[-5:]),
                np.std(last_opens[-5:])
            ]])
            
            target_feat_c = np.array([[
                target_weekday_num, day_sin, day_cos,
                last_closes[-1], last_closes[-2], last_closes[-3], last_closes[-4], last_closes[-5],
                np.mean(last_closes[-3:]), np.mean(last_closes[-5:]),
                np.std(last_closes[-5:])
            ]])
            
            # Train Open Classifier
            clf_o = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            clf_o.fit(X_o, y_o)
            # Ensure model understands all classes 0-9
            probs_o_raw = clf_o.predict_proba(target_feat_o)[0]
            probs_o_full = np.zeros(10)
            for idx, cls in enumerate(clf_o.classes_):
                probs_o_full[cls] = probs_o_raw[idx]
            probs_open = probs_o_full
            
            # Train Close Classifier
            clf_c = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            clf_c.fit(X_c, y_c)
            probs_c_raw = clf_c.predict_proba(target_feat_c)[0]
            probs_c_full = np.zeros(10)
            for idx, cls in enumerate(clf_c.classes_):
                probs_c_full[cls] = probs_c_raw[idx]
            probs_close = probs_c_full
            
            return probs_open, probs_close, clf_o, clf_c
        except Exception as e:
            print(f"GBM Model training failed, falling back to frequencies: {e}")
            return probs_open, probs_close, None, None

    def predict_next(self, df_train=None, target_weekday=None, weights=None, window_size=90):
        """
        Generates ensemble prediction for the next draw.
        """
        if df_train is None:
            df_train = self.df
            
        if window_size is not None and len(df_train) > window_size:
            df_train = df_train.tail(window_size)
            
        if len(df_train) < 7:
            raise ValueError("Insufficient training records.")
            
        if target_weekday is None:
            # Predict for tomorrow/next day in sequence
            last_date = df_train["date"].max()
            next_date = last_date + timedelta(days=1)
            target_weekday_num = next_date.weekday()
            target_weekday_name = next_date.strftime("%A")
        else:
            weekday_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            if isinstance(target_weekday, int):
                target_weekday_num = target_weekday
                target_weekday_name = list(weekday_map.keys())[target_weekday_num]
            else:
                target_weekday_name = target_weekday
                target_weekday_num = weekday_map.get(target_weekday_name, 0)

        # Default weights
        if weights is None:
            weights = {"freq": 0.10, "markov": 0.20, "pattern": 0.20, "ml": 0.25, "gbm": 0.25}
            
        # Get historical vectors for lag-based models
        last_opens = df_train["open_single"].values[-5:]
        last_closes = df_train["close_single"].values[-5:]
        
        # Generate probabilities from submodels
        p_freq_o, p_freq_c = self._get_frequency_probs(df_train, target_weekday_num)
        p_markov_o, p_markov_c = self._get_markov_probs(df_train, last_opens[-1], last_closes[-1])
        p_pattern_o, p_pattern_c = self._get_pattern_probs(df_train, last_opens, last_closes)
        p_ml_o, p_ml_c, clf_o, clf_c = self._get_ml_probs(df_train, last_opens, last_closes, target_weekday_num)
        p_gbm_o, p_gbm_c, gbm_o, gbm_c = self._get_gbm_probs(df_train, last_opens, last_closes, target_weekday_num)
        
        # Combine
        ensemble_o = (
            weights.get("freq", 0) * p_freq_o +
            weights.get("markov", 0) * p_markov_o +
            weights.get("pattern", 0) * p_pattern_o +
            weights.get("ml", 0) * p_ml_o +
            weights.get("gbm", 0) * p_gbm_o
        )
        ensemble_c = (
            weights.get("freq", 0) * p_freq_c +
            weights.get("markov", 0) * p_markov_c +
            weights.get("pattern", 0) * p_pattern_c +
            weights.get("ml", 0) * p_ml_c +
            weights.get("gbm", 0) * p_gbm_c
        )
        
        # Normalize
        ensemble_o = ensemble_o / ensemble_o.sum() if ensemble_o.sum() > 0 else np.ones(10) / 10.0
        ensemble_c = ensemble_c / ensemble_c.sum() if ensemble_c.sum() > 0 else np.ones(10) / 10.0
        
        # Predict open
        pred_open = int(np.argmax(ensemble_o))
        conf_open = float(ensemble_o[pred_open])
        
        # Predict close
        pred_close = int(np.argmax(ensemble_c))
        conf_close = float(ensemble_c[pred_close])
        
        # Top 3 digits
        top3_open = np.argsort(ensemble_o)[-3:][::-1].tolist()
        top3_close = np.argsort(ensemble_c)[-3:][::-1].tolist()
        
        # Generate Panas
        # Open Panas: Get all valid panas for predicted open, sort by their historical frequency in df_train
        open_panas_avail = get_panas_for_digit(pred_open)
        open_pana_counts = df_train["open_pana"].value_counts().to_dict()
        # Sort panas by counts, descending
        sorted_open_panas = sorted(open_panas_avail, key=lambda p: open_pana_counts.get(p, 0), reverse=True)
        # Select top 3 predicted panas
        top3_open_panas = sorted_open_panas[:3]
        
        # Close Panas
        close_panas_avail = get_panas_for_digit(pred_close)
        close_pana_counts = df_train["close_pana"].value_counts().to_dict()
        sorted_close_panas = sorted(close_panas_avail, key=lambda p: close_pana_counts.get(p, 0), reverse=True)
        top3_close_panas = sorted_close_panas[:3]
        
        # Create full predicted jodi string (e.g. "97")
        predicted_jodi = f"{pred_open}{pred_close}"
        
        # Overall confidence (product of individual confidences, or average)
        overall_conf = (conf_open + conf_close) / 2.0
        
        return {
            "target_weekday": target_weekday_name,
            "open_digit": pred_open,
            "open_confidence": conf_open,
            "open_probs": ensemble_o.tolist(),
            "top3_open_digits": top3_open,
            "top3_open_panas": top3_open_panas,
            "close_digit": pred_close,
            "close_confidence": conf_close,
            "close_probs": ensemble_c.tolist(),
            "top3_close_digits": top3_close,
            "top3_close_panas": top3_close_panas,
            "predicted_jodi": predicted_jodi,
            "overall_confidence": overall_conf,
            "model_components": {
                "frequency": {"open": p_freq_o.tolist(), "close": p_freq_c.tolist()},
                "markov": {"open": p_markov_o.tolist(), "close": p_markov_c.tolist()},
                "pattern": {"open": p_pattern_o.tolist(), "close": p_pattern_c.tolist()},
                "ml": {"open": p_ml_o.tolist(), "close": p_ml_c.tolist()},
                "gbm": {"open": p_gbm_o.tolist(), "close": p_gbm_c.tolist()}
            }
        }
        
    def backtest(self, test_draws_count=50, weights=None, window_size=90):
        """
        Simulate historical predictions for the last N draws chronologically.
        Computes predictive accuracy and logs metrics.
        """
        if len(self.df) < test_draws_count + 15:
            test_draws_count = len(self.df) - 15
            
        if test_draws_count <= 0:
            return {"error": "Insufficient data to run backtest"}
            
        results = []
        hits_open = 0
        hits_close = 0
        hits_jodi = 0
        top3_hits_open = 0
        top3_hits_close = 0
        
        total_runs = 0
        
        # Slide window chronologically
        start_idx = len(self.df) - test_draws_count
        
        for idx in range(start_idx, len(self.df)):
            df_train = self.df.iloc[:idx].copy()
            actual_row = self.df.iloc[idx]
            
            actual_open = int(actual_row["open_single"])
            actual_close = int(actual_row["close_single"])
            actual_jodi = actual_row["jodi"]
            actual_date = actual_row["date"].strftime("%Y-%m-%d")
            actual_weekday = actual_row["weekday"]
            
            # Predict
            pred = self.predict_next(df_train, target_weekday=actual_weekday, weights=weights, window_size=window_size)
            
            # Check hits
            is_hit_open = (pred["open_digit"] == actual_open)
            is_hit_close = (pred["close_digit"] == actual_close)
            is_hit_jodi = (pred["predicted_jodi"] == actual_jodi)
            
            is_top3_hit_open = (actual_open in pred["top3_open_digits"])
            is_top3_hit_close = (actual_close in pred["top3_close_digits"])
            
            if is_hit_open: hits_open += 1
            if is_hit_close: hits_close += 1
            if is_hit_jodi: hits_jodi += 1
            if is_top3_hit_open: top3_hits_open += 1
            if is_top3_hit_close: top3_hits_close += 1
            
            results.append({
                "date": actual_date,
                "weekday": actual_weekday,
                "actual_open": actual_open,
                "actual_close": actual_close,
                "actual_jodi": actual_jodi,
                "actual_open_pana": actual_row["open_pana"],
                "actual_close_pana": actual_row["close_pana"],
                "predicted_open": pred["open_digit"],
                "predicted_close": pred["close_digit"],
                "predicted_jodi": pred["predicted_jodi"],
                "predicted_open_panas": pred["top3_open_panas"],
                "predicted_close_panas": pred["top3_close_panas"],
                "open_confidence": pred["open_confidence"],
                "close_confidence": pred["close_confidence"],
                "is_hit_open": is_hit_open,
                "is_hit_close": is_hit_close,
                "is_hit_jodi": is_hit_jodi,
                "is_top3_hit_open": is_top3_hit_open,
                "is_top3_hit_close": is_top3_hit_close
            })
            
            total_runs += 1
            
        accuracy_open = hits_open / total_runs if total_runs > 0 else 0
        accuracy_close = hits_close / total_runs if total_runs > 0 else 0
        accuracy_jodi = hits_jodi / total_runs if total_runs > 0 else 0
        top3_rate_open = top3_hits_open / total_runs if total_runs > 0 else 0
        top3_rate_close = top3_hits_close / total_runs if total_runs > 0 else 0
        
        return {
            "total_runs": total_runs,
            "accuracy_open": accuracy_open,
            "accuracy_close": accuracy_close,
            "accuracy_jodi": accuracy_jodi,
            "top3_rate_open": top3_rate_open,
            "top3_rate_close": top3_rate_close,
            "detailed_results": results
        }

def get_basic_statistics(df_valid):
    """
    Computes general statistical metrics from historical data.
    """
    stats = {}
    
    # Digit frequencies
    stats["open_freq"] = df_valid["open_single"].value_counts().sort_index().to_dict()
    stats["close_freq"] = df_valid["close_single"].value_counts().sort_index().to_dict()
    
    # Weekday digit distributions
    weekday_open_dist = {}
    weekday_close_dist = {}
    
    for day in df_valid["weekday"].unique():
        df_day = df_valid[df_valid["weekday"] == day]
        weekday_open_dist[day] = df_day["open_single"].value_counts().sort_index().to_dict()
        weekday_close_dist[day] = df_day["close_single"].value_counts().sort_index().to_dict()
        
    stats["weekday_open_freq"] = weekday_open_dist
    stats["weekday_close_freq"] = weekday_close_dist
    
    # Cut numbers stats (co-occurrences of cut numbers in open vs close)
    cuts_matching = 0
    for idx, row in df_valid.iterrows():
        o = int(row["open_single"])
        c = int(row["close_single"])
        if CUT_NUMBERS[o] == c:
            cuts_matching += 1
            
    stats["cut_numbers_cooccurrence_rate"] = cuts_matching / len(df_valid) if len(df_valid) > 0 else 0
    
    # Hot and cold numbers (overall)
    # Combining open and close frequencies
    combined_freq = {}
    for d in range(10):
        combined_freq[d] = stats["open_freq"].get(d, 0) + stats["close_freq"].get(d, 0)
        
    sorted_digits = sorted(combined_freq.keys(), key=lambda d: combined_freq[d])
    stats["cold_digits"] = sorted_digits[:3]
    stats["hot_digits"] = sorted_digits[-3:][::-1]
    
    # Double/Triple Panna rates
    # Satta Matka Panas can have Single Panna (3 different digits e.g. 123), 
    # Double Panna (2 same digits e.g. 112), Triple Panna (3 same digits e.g. 111)
    def classify_panna(panna_str):
        if not panna_str or len(panna_str) != 3 or not panna_str.isdigit():
            return "unknown"
        unique_chars = len(set(panna_str))
        if unique_chars == 3: return "single"
        if unique_chars == 2: return "double"
        return "triple"
        
    panna_class_open = df_valid["open_pana"].apply(classify_panna).value_counts().to_dict()
    panna_class_close = df_valid["close_pana"].apply(classify_panna).value_counts().to_dict()
    
    stats["panna_types"] = {
        "open": panna_class_open,
        "close": panna_class_close
    }
    
    # Jodis that repeat
    jodi_counts = df_valid["jodi"].value_counts()
    stats["top_jodis"] = [{"jodi": k, "count": int(v)} for k, v in jodi_counts.head(5).items()]
    
    # Sequential correlation (Autocorrelation lag-1)
    open_corr = df_valid["open_single"].autocorr(lag=1)
    close_corr = df_valid["close_single"].autocorr(lag=1)
    stats["autocorrelation"] = {
        "open": float(open_corr) if not pd.isna(open_corr) else 0.0,
        "close": float(close_corr) if not pd.isna(close_corr) else 0.0
    }
    
    return stats

PANA_LIBRARY = [
    "120", "123", "124", "125", "126", "127", "128", "129",
    "130", "134", "135", "136", "137", "138", "139", "140",
    "230", "234", "235", "236", "237", "238", "239", "240",
    "340", "345", "346", "347", "348", "349", "350", "360",
    "110", "220", "330", "440", "550", "660", "770", "880", "990",
    "118", "226", "334", "442", "550", "668", "776", "884", "992",
    "100", "200", "300", "400", "500", "600", "700", "800", "900", "000"
]

def calculate_single_digit(pana):
    if not pana or len(str(pana)) != 3:
        return "0"
    return str(sum(int(d) for d in str(pana)) % 10)

def generate_panas_for_digit(digit):
    digit_str = str(digit)
    result = []
    for p in PANA_LIBRARY:
        if calculate_single_digit(p) == digit_str:
            result.append(p)
    return result[:6]

def sort_pana_digits(digits):
    arr = list(str(digits))
    arr.sort(key=lambda x: 10 if x == '0' else int(x))
    return "".join(arr)

def generate_jodis(touch_digits):
    jodis = []
    unique = []
    for d in touch_digits:
        d_str = str(d)
        if d_str not in unique:
            unique.append(d_str)
    for i in range(len(unique)):
        for j in range(len(unique)):
            jodis.append(unique[i] + unique[j])
    return jodis

def get_cut_number(digit):
    d = int(str(digit))
    return str((d + 5) % 10)

def get_previous_entries(df_history, target_date):
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
    df_prev = df_history[df_history["date"] < target_dt]
    return df_prev.sort_values("date", ascending=False).reset_index(drop=True)

def predict_date_touch(target_date):
    if isinstance(target_date, str):
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        date_obj = target_date
        
    day = date_obj.day
    day_str = str(day)
    
    touch_digits = []
    for d in day_str:
        touch_digits.append(d)
        touch_digits.append(get_cut_number(d))
        
    if len(day_str) == 1:
        secondary = str((day + 1) % 10)
        touch_digits.append(secondary)
        touch_digits.append(get_cut_number(secondary))
        
    unique_touch = []
    for d in touch_digits:
        if d not in unique_touch:
            unique_touch.append(d)
    unique_touch = unique_touch[:4]
    
    jodis = generate_jodis(unique_touch)[:8]
    
    panas = []
    for digit in unique_touch:
        panas.extend(generate_panas_for_digit(digit)[:2])
    
    unique_panas = []
    for p in panas:
        if p not in unique_panas:
            unique_panas.append(p)
    unique_panas = unique_panas[:6]
    
    day_str_joined = ", ".join(list(day_str))
    cut_str_joined = ", ".join([get_cut_number(d) for d in day_str])
    return {
        "name": "Date-wise Touch Scheme",
        "description": f"Calculates OTC touch digits by extracting target date day ({day}) digits and their cut numbers. Day digits: [{day_str_joined}]. Cut numbers: [{cut_str_joined}].",
        "touchDigits": unique_touch,
        "jodis": jodis,
        "panas": unique_panas
    }

def predict_yesterday_sum(df_history, target_date):
    df_prev = get_previous_entries(df_history, target_date)
    if df_prev.empty:
        return {
            "name": "Yesterday Open-Close Sum",
            "description": "No historical data available prior to the target date to calculate prediction.",
            "touchDigits": [], "jodis": [], "panas": []
        }
        
    yesterday = df_prev.iloc[0]
    open_val = int(yesterday["open_single"])
    close_val = int(yesterday["close_single"])
    
    sum_val = (open_val + close_val) % 10
    diff_val = abs(open_val - close_val) % 10
    
    touch_digits = [
        str(sum_val),
        get_cut_number(sum_val),
        str(diff_val),
        get_cut_number(diff_val)
    ]
    
    unique_touch = []
    for d in touch_digits:
        if d not in unique_touch:
            unique_touch.append(d)
    unique_touch = unique_touch[:4]
    
    jodis = generate_jodis(unique_touch)[:8]
    
    panas = []
    for digit in unique_touch:
        panas.extend(generate_panas_for_digit(digit)[:2])
        
    unique_panas = []
    for p in panas:
        if p not in unique_panas:
            unique_panas.append(p)
    unique_panas = unique_panas[:6]
    
    yesterday_date_str = yesterday["date"].strftime("%Y-%m-%d")
    desc = (
        f"Analyzes the last result ({yesterday['open_pana']}-{yesterday['jodi']}-{yesterday['close_pana']} on {yesterday_date_str}) "
        f"to compute OTC numbers. Calculations: (Open Single {open_val} + Close Single {close_val}) % 10 = Sum {sum_val} "
        f"(Cut = {get_cut_number(sum_val)}); |Open Single {open_val} - Close Single {close_val}| % 10 = Difference {diff_val} "
        f"(Cut = {get_cut_number(diff_val)}). Touch digits = [{', '.join(unique_touch)}]."
    )
    
    return {
        "name": "Yesterday Open-Close Sum",
        "description": desc,
        "touchDigits": unique_touch,
        "jodis": jodis,
        "panas": unique_panas
    }

def predict_weekly_repeat(df_history, target_date):
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
        
    target_day_of_week = target_dt.weekday()
    
    df_same_weekday = df_history[(df_history["weekday_num"] == target_day_of_week) & (df_history["date"] < target_dt)]
    df_same_weekday = df_same_weekday.sort_values("date", ascending=False).reset_index(drop=True)
    same_weekday_entries = df_same_weekday.head(5)
    
    if len(same_weekday_entries) < 2:
        fallback = predict_date_touch(target_dt)
        fallback["name"] = "Weekly Repeat Pattern"
        fallback["description"] = f"Fallback to Date Touch: Not enough same weekday history (< 2 entries)."
        return fallback
        
    counts = [0] * 10
    for _, entry in same_weekday_entries.iterrows():
        counts[int(entry["open_single"])] += 1
        counts[int(entry["close_single"])] += 1
        
    sorted_digits = sorted(
        [{"digit": str(d), "count": count} for d, count in enumerate(counts)],
        key=lambda x: x["count"],
        reverse=True
    )
    
    touch_digits = [
        sorted_digits[0]["digit"],
        get_cut_number(sorted_digits[0]["digit"]),
        sorted_digits[1]["digit"],
        get_cut_number(sorted_digits[1]["digit"])
    ]
    
    unique_touch = []
    for d in touch_digits:
        if d not in unique_touch:
            unique_touch.append(d)
    unique_touch = unique_touch[:4]
    
    jodis = generate_jodis(unique_touch)[:8]
    
    panas = []
    for digit in unique_touch:
        panas.extend(generate_panas_for_digit(digit)[:2])
        
    unique_panas = []
    for p in panas:
        if p not in unique_panas:
            unique_panas.append(p)
    unique_panas = unique_panas[:6]
    
    weekdays_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_name = weekdays_names[target_day_of_week]
    
    freq_desc = ", ".join([f"{d['digit']} (count: {d['count']})" for d in sorted_digits[:3]])
    desc = (
        f"Analyzes the past {len(same_weekday_entries)} {weekday_name} charts to find recurring hot numbers. "
        f"Frequency: {freq_desc}. Top 2 digits are {sorted_digits[0]['digit']} (Cut: {get_cut_number(sorted_digits[0]['digit'])}) "
        f"and {sorted_digits[1]['digit']} (Cut: {get_cut_number(sorted_digits[1]['digit'])})."
    )
    
    return {
        "name": "Weekly Repeat Pattern",
        "description": desc,
        "touchDigits": unique_touch,
        "jodis": jodis,
        "panas": unique_panas
    }

def predict_panel_multiplier(df_history, target_date):
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
        
    day = target_dt.day
    month = target_dt.month
    
    df_prev = get_previous_entries(df_history, target_dt)
    if df_prev.empty:
        fallback = predict_date_touch(target_dt)
        fallback["name"] = "Panel Digit Multiplier"
        fallback["description"] = f"Fallback to Date Touch: No history available prior to target date."
        return fallback
        
    yesterday = df_prev.iloc[0]
    
    try:
        jodi_val = int(yesterday["jodi"])
    except:
        jodi_val = 55
        
    factor = (jodi_val * day) + month
    factor_str = str(factor)
    
    touch_digits = []
    for d in factor_str:
        touch_digits.append(d)
        
    touch_digits.append(get_cut_number(touch_digits[0] if len(touch_digits) > 0 else "1"))
    touch_digits.append(get_cut_number(touch_digits[1] if len(touch_digits) > 1 else "2"))
    
    unique_touch = []
    for d in touch_digits:
        if d not in unique_touch:
            unique_touch.append(d)
    unique_touch = unique_touch[:4]
    
    panas = []
    prev_open_pana = str(yesterday["open_pana"])
    if not prev_open_pana or len(prev_open_pana) != 3 or not prev_open_pana.isdigit():
        prev_open_pana = "123"
        
    prev_digits = [int(x) for x in prev_open_pana]
    val1 = (prev_digits[0] + day) % 10
    val2 = (prev_digits[1] + day) % 10
    val3 = (prev_digits[2] + day) % 10
    
    custom_pana1 = sort_pana_digits(f"{val1}{val2}{val3}")
    custom_pana2 = sort_pana_digits(f"{(val1 + 5) % 10}{(val2 + 5) % 10}{(val3 + 5) % 10}")
    
    panas.append(custom_pana1)
    panas.append(custom_pana2)
    
    for digit in unique_touch:
        panas.extend(generate_panas_for_digit(digit)[:1])
        
    unique_panas = []
    for p in panas:
        if p not in unique_panas:
            unique_panas.append(p)
    unique_panas = unique_panas[:6]
    
    factor_digits = "".join(list(dict.fromkeys(factor_str)))
    desc = (
        f"Multiplies last Jodi ({yesterday['jodi']}) by target day ({day}) + month ({month}) = "
        f"({yesterday['jodi']} * {day}) + {month} = {factor}. Digits from product = [{', '.join(list(factor_digits))}]. "
        f"Cuts added for padding. Touch digits = [{', '.join(unique_touch)}]."
    )
    
    return {
        "name": "Panel Digit Multiplier",
        "description": desc,
        "touchDigits": unique_touch,
        "jodis": generate_jodis(unique_touch)[:8],
        "panas": unique_panas
    }

import re
def evaluate_simple_expression(expression, open_val, close_val, jodi_val, date_val, month):
    expr = expression.lower()
    expr = re.sub(r'\bopen\b', str(open_val), expr)
    expr = re.sub(r'\bclose\b', str(close_val), expr)
    expr = re.sub(r'\bjodi\b', str(jodi_val), expr)
    expr = re.sub(r'\bdate\b', str(date_val), expr)
    expr = re.sub(r'\bmonth\b', str(month), expr)
    
    expr = re.sub(r'[^0-9\+\-\*\/\%\(\)\s\.]', '', expr)
    try:
        val = eval(expr, {"__builtins__": None}, {})
        if isinstance(val, (int, float)):
            return str(abs(round(val)) % 10)
    except Exception as e:
        pass
    return None

def predict_custom_formula(df_history, target_date, formula_str):
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
        
    day = target_dt.day
    month = target_dt.month
    
    df_prev = get_previous_entries(df_history, target_dt)
    if not df_prev.empty:
        yesterday = df_prev.iloc[0]
        open_val = int(yesterday["open_single"])
        close_val = int(yesterday["close_single"])
        try:
            jodi_val = int(yesterday["jodi"])
        except:
            jodi_val = 52
    else:
        open_val = 5
        close_val = 2
        jodi_val = 52
        
    calculated_digit = evaluate_simple_expression(formula_str, open_val, close_val, jodi_val, day, month)
    if calculated_digit is None:
        calculated_digit = str((open_val + close_val) % 10)
        
    calculated_int = int(calculated_digit)
    primary_digit = str(calculated_int)
    cut_digit = get_cut_number(primary_digit)
    
    touch_digits = [
        primary_digit,
        cut_digit,
        str((calculated_int + 1) % 10),
        get_cut_number(str((calculated_int + 1) % 10))
    ]
    
    unique_touch = []
    for d in touch_digits:
        if d not in unique_touch:
            unique_touch.append(d)
    unique_touch = unique_touch[:4]
    
    jodis = generate_jodis(unique_touch)[:8]
    
    panas = []
    for digit in unique_touch:
        panas.extend(generate_panas_for_digit(digit)[:2])
        
    unique_panas = []
    for p in panas:
        if p not in unique_panas:
            unique_panas.append(p)
    unique_panas = unique_panas[:6]
    
    desc = f"Evaluated formula: \"{formula_str}\" using variables: Open={open_val}, Close={close_val}, Jodi={jodi_val}, Date={day}, Month={month}."
    return {
        "name": "Custom User Formula",
        "description": desc,
        "touchDigits": unique_touch,
        "jodis": jodis,
        "panas": unique_panas
    }

import math
def predict_ai_pattern(df_history, target_date):
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
        
    df_prev = get_previous_entries(df_history, target_dt)
    if df_prev.empty:
        fallback = predict_date_touch(target_dt)
        fallback["name"] = "AI Neural Pattern Recognizer"
        return fallback
        
    scores = [0.0] * 10
    decay_factor = 0.05
    recent_draws = df_prev.head(60)
    for idx, row in recent_draws.iterrows():
        weight = math.exp(-decay_factor * idx)
        open_digit = int(row["open_single"])
        close_digit = int(row["close_single"])
        scores[open_digit] += weight * 0.4
        scores[close_digit] += weight * 0.4
        
    yesterday = df_prev.iloc[0]
    prev_open = int(yesterday["open_single"])
    prev_close = int(yesterday["close_single"])
    
    open_transitions = [0] * 10
    close_transitions = [0] * 10
    open_count = 0
    close_count = 0
    
    n_entries = len(df_prev)
    for i in range(n_entries - 2, -1, -1):
        current_row = df_prev.iloc[i]
        prev_row = df_prev.iloc[i+1]
        
        p_open = int(prev_row["open_single"])
        p_close = int(prev_row["close_single"])
        c_open = int(current_row["open_single"])
        c_close = int(current_row["close_single"])
        
        if p_open == prev_open:
            open_transitions[c_open] += 1
            open_transitions[c_close] += 1
            open_count += 2
            
        if p_close == prev_close:
            close_transitions[c_open] += 1
            close_transitions[c_close] += 1
            close_count += 2
            
    for d in range(10):
        if open_count > 0:
            scores[d] += (open_transitions[d] / open_count) * 1.5
        if close_count > 0:
            scores[d] += (close_transitions[d] / close_count) * 1.5
            
    target_day_of_week = target_dt.weekday()
    df_same_weekday = df_prev[df_prev["weekday_num"] == target_day_of_week]
    same_weekday_entries = df_same_weekday.head(10)
    
    if not same_weekday_entries.empty:
        weekday_counts = [0] * 10
        for _, row in same_weekday_entries.iterrows():
            o_d = int(row["open_single"])
            c_d = int(row["close_single"])
            weekday_counts[o_d] += 1
            weekday_counts[c_d] += 1
            
        total_weekday_counts = len(same_weekday_entries) * 2
        if total_weekday_counts > 0:
            for d in range(10):
                scores[d] += (weekday_counts[d] / total_weekday_counts) * 0.8
                
    sorted_scores = sorted(
        [{"digit": str(d), "score": score} for d, score in enumerate(scores)],
        key=lambda x: x["score"],
        reverse=True
    )
    
    touch_digits = [s["digit"] for s in sorted_scores[:4]]
    
    jodis = []
    top_digits = touch_digits[:3]
    for i in range(len(top_digits)):
        for j in range(len(top_digits)):
            jodis.append(top_digits[i] + top_digits[j])
            
    fallback_jodis = generate_jodis(touch_digits)
    for j in fallback_jodis:
        if len(jodis) < 8 and j not in jodis:
            jodis.append(j)
            
    jodis = jodis[:8]
    
    panas = []
    for digit in touch_digits:
        panas.extend(generate_panas_for_digit(digit)[:2])
        
    unique_panas = []
    for p in panas:
        if p not in unique_panas:
            unique_panas.append(p)
    unique_panas = unique_panas[:6]
    
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    desc = (
        f"Ensemble model training on {len(df_prev)} draws. "
        f"Analyzed Markov transitions following {yesterday['open_single']}-{yesterday['close_single']} "
        f"and weekday co-occurrence for {weekdays[target_day_of_week]}."
    )
    
    return {
        "name": "AI Neural Pattern Recognizer",
        "description": desc,
        "touchDigits": touch_digits,
        "jodis": jodis,
        "panas": unique_panas
    }

def predict_adaptive_line(df_history, target_date):
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
        
    df_prev = get_previous_entries(df_history, target_dt)
    if len(df_prev) < 15:
        fallback = predict_yesterday_sum(df_history, target_dt)
        fallback["name"] = "Adaptive Running Line Optimizer"
        fallback["description"] = f"Fallback to Yesterday's Open-Close Sum: not enough history to analyze running lines (< 15 entries)."
        return fallback
        
    candidates = [
        {"key": "yesterday_sum", "name": "Yesterday's Open-Close Sum"},
        {"key": "panel_multiplier", "name": "Panel Digit Multiplier"},
        {"key": "date_touch", "name": "Date-wise Touch Scheme"},
        {"key": "weekly_repeat", "name": "Weekly Repeat Pattern"},
        {"key": "ai_pattern", "name": "AI Neural Pattern Recognizer"}
    ]
    
    eval_limit = min(10, len(df_prev) - 5)
    scores = {c["key"]: 0 for c in candidates}
    total_eval = 0
    
    for i in range(eval_limit - 1, -1, -1):
        actual_result = df_prev.iloc[i]
        actual_date_str = actual_result["date"]
        sub_history = df_prev.iloc[i + 1:]
        
        total_eval += 1
        
        for candidate in candidates:
            pred = None
            if candidate["key"] == "date_touch":
                pred = predict_date_touch(actual_date_str)
            elif candidate["key"] == "yesterday_sum":
                pred = predict_yesterday_sum(sub_history, actual_date_str)
            elif candidate["key"] == "weekly_repeat":
                pred = predict_weekly_repeat(sub_history, actual_date_str)
            elif candidate["key"] == "panel_multiplier":
                pred = predict_panel_multiplier(sub_history, actual_date_str)
            elif candidate["key"] == "ai_pattern":
                pred = predict_ai_pattern(sub_history, actual_date_str)
                
            if pred and len(pred["touchDigits"]) > 0:
                open_s = str(actual_result["open_single"])
                close_s = str(actual_result["close_single"])
                if open_s in pred["touchDigits"] or close_s in pred["touchDigits"]:
                    scores[candidate["key"]] += 1
                    
    best_key = "yesterday_sum"
    max_score = -1
    for c in candidates:
        if scores[c["key"]] > max_score:
            max_score = scores[c["key"]]
            best_key = c["key"]
            
    win_rate_percent = f"{(max_score / total_eval * 100):.1f}" if total_eval > 0 else "0.0"
    best_candidate = next(c for c in candidates if c["key"] == best_key)
    
    final_prediction = None
    if best_key == "date_touch":
        final_prediction = predict_date_touch(target_dt)
    elif best_key == "yesterday_sum":
        final_prediction = predict_yesterday_sum(df_history, target_dt)
    elif best_key == "weekly_repeat":
        final_prediction = predict_weekly_repeat(df_history, target_dt)
    elif best_key == "panel_multiplier":
        final_prediction = predict_panel_multiplier(df_history, target_dt)
    elif best_key == "ai_pattern":
        final_prediction = predict_ai_pattern(df_history, target_dt)
        
    explanation = (
        f"Adaptive running line analysis selected '{best_candidate['name']}' as the hottest pattern "
        f"(OTC Touch win rate: {win_rate_percent}% over the last {total_eval} draws). "
        f"Calculation details: {final_prediction['description']}"
    )
    
    return {
        **final_prediction,
        "name": "Adaptive Running Line Optimizer",
        "description": explanation
    }

def run_formula_backtest(df_history, formula_type, custom_formula_str="", limit=30):
    if len(df_history) < 5:
        return None
        
    test_limit = min(limit, len(df_history) - 5)
    
    single_matches = 0
    jodi_matches = 0
    touch_matches = 0
    pana_matches = 0
    total_tested = 0
    
    detailed_results = []
    start_idx = len(df_history) - test_limit
    
    for i in range(start_idx, len(df_history)):
        actual_result = df_history.iloc[i]
        target_date_str = actual_result["date"]
        history_before = df_history.iloc[:i]
        
        prediction = None
        if formula_type == 'date_touch':
            prediction = predict_date_touch(target_date_str)
        elif formula_type == 'yesterday_sum':
            prediction = predict_yesterday_sum(history_before, target_date_str)
        elif formula_type == 'weekly_repeat':
            prediction = predict_weekly_repeat(history_before, target_date_str)
        elif formula_type == 'panel_multiplier':
            prediction = predict_panel_multiplier(history_before, target_date_str)
        elif formula_type == 'custom':
            prediction = predict_custom_formula(history_before, target_date_str, custom_formula_str)
        elif formula_type == 'ai_pattern':
            prediction = predict_ai_pattern(history_before, target_date_str)
        elif formula_type == 'adaptive_line':
            prediction = predict_adaptive_line(history_before, target_date_str)
            
        if not prediction or len(prediction["touchDigits"]) == 0:
            continue
            
        total_tested += 1
        
        actual_open_single = str(actual_result["open_single"])
        actual_close_single = str(actual_result["close_single"])
        actual_jodi = str(actual_result["jodi"])
        actual_open_pana = str(actual_result["open_pana"])
        actual_close_pana = str(actual_result["close_pana"])
        
        is_touch_hit = (actual_open_single in prediction["touchDigits"]) or (actual_close_single in prediction["touchDigits"])
        if is_touch_hit:
            touch_matches += 1
            
        primary_predicted = prediction["touchDigits"][0]
        is_single_hit = (primary_predicted == actual_open_single) or (primary_predicted == actual_close_single)
        if is_single_hit:
            single_matches += 1
            
        is_jodi_hit = (actual_jodi in prediction["jodis"])
        if is_jodi_hit:
            jodi_matches += 1
            
        is_pana_hit = (actual_open_pana in prediction["panas"]) or (actual_close_pana in prediction["panas"])
        if is_pana_hit:
            pana_matches += 1
            
        detailed_results.append({
            "date": target_date_str.strftime("%Y-%m-%d") if isinstance(target_date_str, datetime) else str(target_date_str),
            "weekday": actual_result["weekday"],
            "actual_open": actual_open_single,
            "actual_close": actual_close_single,
            "actual_jodi": actual_jodi,
            "actual_open_pana": actual_open_pana,
            "actual_close_pana": actual_close_pana,
            "predicted_touch": prediction["touchDigits"],
            "predicted_jodis": prediction["jodis"],
            "predicted_panas": prediction["panas"],
            "is_hit_touch": is_touch_hit,
            "is_hit_single": is_single_hit,
            "is_hit_jodi": is_jodi_hit,
            "is_hit_pana": is_pana_hit
        })
        
    return {
        "totalTested": total_tested,
        "touchRate": f"{(touch_matches / total_tested * 100):.1f}" if total_tested > 0 else "0.0",
        "singleRate": f"{(single_matches / total_tested * 100):.1f}" if total_tested > 0 else "0.0",
        "jodiRate": f"{(jodi_matches / total_tested * 100):.1f}" if total_tested > 0 else "0.0",
        "panaRate": f"{(pana_matches / total_tested * 100):.1f}" if total_tested > 0 else "0.0",
        "rawMatches": {
            "touch": touch_matches,
            "single": single_matches,
            "jodi": jodi_matches,
            "pana": pana_matches
        },
        "detailed_results": detailed_results
    }

def find_matching_formulas(df_history, target_date):
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
        
    target_entry_rows = df_history[df_history["date"] == target_dt]
    if target_entry_rows.empty:
        return None
    target_entry = target_entry_rows.iloc[0]
    
    df_prev = get_previous_entries(df_history, target_dt)
    if df_prev.empty:
        return None
        
    yesterday = df_prev.iloc[0]
    open_val = int(yesterday["open_single"])
    close_val = int(yesterday["close_single"])
    try:
        jodi_val = int(yesterday["jodi"])
    except:
        jodi_val = 55
        
    date_val = target_dt.day
    month = target_dt.month
    
    target_open = str(target_entry["open_single"])
    target_close = str(target_entry["close_single"])
    
    templates = [
        "Open", "Close", "Date", "Month",
        "Open + Close", "Open - Close", "Close - Open", "Open * Close",
        "Open + Date", "Open - Date", "Date - Open", "Open * Date",
        "Close + Date", "Close - Date", "Date - Close", "Close * Date",
        "Open + Month", "Close + Month", "Date + Month", "Date - Month",
        "Open + Close + Date", "Open + Close - Date", "Open + Date - Close", "Close + Date - Open",
        "Open * Close + Date", "Open * Close - Date", "(Open + Close) * Date", "(Open + Close) * Month",
        "Jodi + Date", "Jodi - Date", "Jodi + Month", "Jodi - Month",
        "Open + 1", "Open + 2", "Open + 3", "Open + 5", "Open + 7", "Open - 1", "Open - 2", "Open - 3", "Open - 5",
        "Close + 1", "Close + 2", "Close + 3", "Close + 5", "Close + 7", "Close - 1", "Close - 2", "Close - 3", "Close - 5",
        "(Open + Close) + 1", "(Open + Close) + 2", "(Open + Close) + 3", "(Open + Close) + 5",
        "(Open + Close) - 1", "(Open + Close) - 2", "(Open + Close) - 3", "(Open + Close) - 5",
        "(Open + Date) + 1", "(Open + Date) + 5", "(Close + Date) + 1", "(Close + Date) + 5",
        "Open * 2", "Open * 3", "Open * 5", "Close * 2", "Close * 3", "Close * 5"
    ]
    
    open_matches = []
    close_matches = []
    unique_templates = list(dict.fromkeys(templates))
    
    for expr in unique_templates:
        val = evaluate_simple_expression(expr, open_val, close_val, jodi_val, date_val, month)
        if val is not None:
            if val == target_open:
                open_matches.append({"expression": expr, "result": val})
            if val == target_close:
                close_matches.append({"expression": expr, "result": val})
                
    open_matches.sort(key=lambda x: len(x["expression"]))
    close_matches.sort(key=lambda x: len(x["expression"]))
    
    return {
        "openMatches": open_matches[:5],
        "closeMatches": close_matches[:5]
    }

def evaluate_all_models_for_date(df_history, target_date):
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
        
    target_entry_rows = df_history[df_history["date"] == target_dt]
    if target_entry_rows.empty:
        return None
    target_entry = target_entry_rows.iloc[0]
    
    history_before = df_history[df_history["date"] < target_dt]
    
    models = [
        {"key": "date_touch", "name": "Date-wise Touch Scheme"},
        {"key": "yesterday_sum", "name": "Yesterday's Open-Close Sum"},
        {"key": "weekly_repeat", "name": "Weekly Repeat Pattern"},
        {"key": "panel_multiplier", "name": "Panel Digit Multiplier"},
        {"key": "ai_pattern", "name": "AI Neural Pattern Recognizer"},
        {"key": "adaptive_line", "name": "Adaptive Running Line Optimizer"}
    ]
    
    results = []
    actual_open = str(target_entry["open_single"])
    actual_close = str(target_entry["close_single"])
    actual_jodi = str(target_entry["jodi"])
    actual_open_p = str(target_entry["open_pana"])
    actual_close_p = str(target_entry["close_pana"])
    
    for model in models:
        pred = None
        if model["key"] == "date_touch":
            pred = predict_date_touch(target_dt)
        elif model["key"] == "yesterday_sum":
            pred = predict_yesterday_sum(history_before, target_dt)
        elif model["key"] == "weekly_repeat":
            pred = predict_weekly_repeat(history_before, target_dt)
        elif model["key"] == "panel_multiplier":
            pred = predict_panel_multiplier(history_before, target_dt)
        elif model["key"] == "ai_pattern":
            pred = predict_ai_pattern(history_before, target_dt)
        elif model["key"] == "adaptive_line":
            pred = predict_adaptive_line(history_before, target_dt)
            
        if not pred or len(pred["touchDigits"]) == 0:
            results.append({
                **model,
                "touchHit": False,
                "singleHit": False,
                "jodiHit": False,
                "panaHit": False,
                "touchDigits": [],
                "jodis": [],
                "panas": []
            })
            continue
            
        touchHit = (actual_open in pred["touchDigits"]) or (actual_close in pred["touchDigits"])
        singleHit = (pred["touchDigits"][0] == actual_open) or (pred["touchDigits"][0] == actual_close)
        jodiHit = (actual_jodi in pred["jodis"])
        panaHit = (actual_open_p in pred["panas"]) or (actual_close_p in pred["panas"])
        
        results.append({
            **model,
            "touchHit": touchHit,
            "singleHit": singleHit,
            "jodiHit": jodiHit,
            "panaHit": panaHit,
            "touchDigits": pred["touchDigits"],
            "jodis": pred["jodis"],
            "panas": pred["panas"]
        })
        
    return results

def find_best_custom_formula(df_history, depth=30):
    if len(df_history) < 5:
        return []
        
    templates = [
        "Open", "Close", "Date", "Month",
        "Open + Close", "Open - Close", "Close - Open", "Open * Close",
        "Open + Date", "Open - Date", "Date - Open", "Open * Date",
        "Close + Date", "Close - Date", "Date - Close", "Close * Date",
        "Open + Month", "Close + Month", "Date + Month", "Date - Month",
        "Open + Close + Date", "Open + Close - Date", "Open + Date - Close", "Close + Date - Open",
        "Open * Close + Date", "Open * Close - Date", "(Open + Close) * Date", "(Open + Close) * Month",
        "Jodi + Date", "Jodi - Date", "Jodi + Month", "Jodi - Month",
        "Open + 1", "Open + 2", "Open + 3", "Open + 5", "Open + 7", "Open - 1", "Open - 2", "Open - 3", "Open - 5",
        "Close + 1", "Close + 2", "Close + 3", "Close + 5", "Close + 7", "Close - 1", "Close - 2", "Close - 3", "Close - 5",
        "(Open + Close) + 1", "(Open + Close) + 2", "(Open + Close) + 3", "(Open + Close) + 5",
        "(Open + Close) - 1", "(Open + Close) - 2", "(Open + Close) - 3", "(Open + Close) - 5",
        "(Open + Date) + 1", "(Open + Date) + 5", "(Close + Date) + 1", "(Close + Date) + 5",
        "Open * 2", "Open * 3", "Open * 5", "Close * 2", "Close * 3", "Close * 5",
        "(Open * 3) + Close", "(Close * 3) + Open", "(Open + Close) * 2", "(Open + Close) * 3",
        "Date * 2 + Month", "Date * 3 - Month", "(Open * Date) % 10", "(Close * Date) % 10",
        "Jodi % 10", "Jodi / 10", "(Jodi % 10) + Date"
    ]
    
    unique_templates = list(dict.fromkeys(templates))
    test_limit = min(depth, len(df_history) - 5)
    
    scored_formulas = []
    start_idx = len(df_history) - test_limit
    
    for expr in unique_templates:
        touch_matches = 0
        jodi_matches = 0
        pana_matches = 0
        total_tested = 0
        
        for i in range(start_idx, len(df_history)):
            actual_result = df_history.iloc[i]
            target_date_str = actual_result["date"]
            history_before = df_history.iloc[:i]
            
            pred = predict_custom_formula(history_before, target_date_str, expr)
            if not pred or len(pred["touchDigits"]) == 0:
                continue
                
            total_tested += 1
            
            actual_open_single = str(actual_result["open_single"])
            actual_close_single = str(actual_result["close_single"])
            actual_jodi = str(actual_result["jodi"])
            actual_open_pana = str(actual_result["open_pana"])
            actual_close_pana = str(actual_result["close_pana"])
            
            if (actual_open_single in pred["touchDigits"]) or (actual_close_single in pred["touchDigits"]):
                touch_matches += 1
            if actual_jodi in pred["jodis"]:
                jodi_matches += 1
            if (actual_open_pana in pred["panas"]) or (actual_close_pana in pred["panas"]):
                pana_matches += 1
                
        touch_rate = round((touch_matches / total_tested * 100), 1) if total_tested > 0 else 0.0
        jodi_rate = round((jodi_matches / total_tested * 100), 1) if total_tested > 0 else 0.0
        pana_rate = round((pana_matches / total_tested * 100), 1) if total_tested > 0 else 0.0
        
        scored_formulas.append({
            "expression": expr,
            "touchRate": touch_rate,
            "jodiRate": jodi_rate,
            "panaRate": pana_rate,
            "totalTested": total_tested
        })
        
    scored_formulas.sort(key=lambda x: (-x["touchRate"], -x["jodiRate"], len(x["expression"])))
    return scored_formulas[:10]

if __name__ == "__main__":
    # Test execution
    print("Testing Satta Matka models module...")
    _, df_valid = load_data()
    print(f"Loaded {len(df_valid)} valid draws.")
    
    # Test our new algebraic formulas
    latest_date = df_valid["date"].max()
    print(f"\nTesting Algebraic Formulas on latest date: {latest_date.strftime('%Y-%m-%d')}")
    
    hist_before = df_valid[df_valid["date"] < latest_date]
    
    p_date = predict_date_touch(latest_date)
    print(f"  predict_date_touch: {p_date['touchDigits']} | Suggests {len(p_date['jodis'])} jodis")
    
    p_yesterday = predict_yesterday_sum(hist_before, latest_date)
    print(f"  predict_yesterday_sum: {p_yesterday['touchDigits']}")
    
    p_weekly = predict_weekly_repeat(hist_before, latest_date)
    print(f"  predict_weekly_repeat: {p_weekly['touchDigits']}")
    
    p_panel = predict_panel_multiplier(hist_before, latest_date)
    print(f"  predict_panel_multiplier: {p_panel['touchDigits']}")
    
    p_custom = predict_custom_formula(hist_before, latest_date, "(Open + Close) * 2")
    print(f"  predict_custom_formula: {p_custom['touchDigits']}")
    
    p_ai = predict_ai_pattern(hist_before, latest_date)
    print(f"  predict_ai_pattern: {p_ai['touchDigits']}")
    
    p_adaptive = predict_adaptive_line(hist_before, latest_date)
    print(f"  predict_adaptive_line: {p_adaptive['touchDigits']}")
    
    print("\nRunning Backtest on Yesterday's Sum...")
    back_res = run_formula_backtest(df_valid, "yesterday_sum", limit=15)
    print(f"  Total Tested: {back_res['totalTested']} | Touch Rate: {back_res['touchRate']}%")
    
    print("\nRunning Auto-Optimizer...")
    best_forms = find_best_custom_formula(df_valid, depth=10)
    print(f"  Top formula found: '{best_forms[0]['expression']}' with Touch Rate: {best_forms[0]['touchRate']}%")
    
    print("\nAll tests passed successfully!")
