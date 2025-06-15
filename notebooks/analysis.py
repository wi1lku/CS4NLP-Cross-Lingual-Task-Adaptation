import json
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import lang2vec.lang2vec  as l2v
import lang2vec
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import math

code2name = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'cs': 'Czech',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'fi': 'Finnish',
    'fr': 'French',
    'gl': 'Galician',
    'hi': 'Hindi',
    'id': 'Indonesian',
    'is': 'Icelandic',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'th': 'Thai',
    'tr': 'Turkish',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese'
}

code2color = {
    'ar': '#e6194b',
    'bg': '#3cb44b',
    'cs': '#ffe119',
    'de': '#0082c8',
    'el': '#f58231',
    'en': '#911eb4',
    'es': '#46f0f0',
    'fi': '#f032e6',
    'fr': '#000080',
    'gl': '#fabebe',
    'hi': '#008080',
    'id': '#e6beff',
    'is': '#aa6e28',
    'it': '#800000',
    'ja': '#aaffc3',
    'ko': '#808000',
    'pl': '#ff88aa',
    'pt': '#d2f53c',
    'ru': '#808080',
    'sv': "#0cf63f",
    'sw': '#9a6324',
    'th': '#4363d8',
    'tr': '#ffe0b3',
    'ur': '#bfef45',
    'vi': '#fabed4',
    'zh': '#a9a9a9'
}

language_code_map = {
    'ar': 'arb',  # Arabic
    'bg': 'bul',  # Bulgarian
    'de': 'deu',  # German
    'el': 'ell',  # Greek
    'en': 'eng',  # English
    'es': 'spa',  # Spanish
    'fr': 'fra',  # French
    'hi': 'hin',  # Hindi
    'ru': 'rus',  # Russian
    'sw': 'swa',  # Swahili
    'th': 'tha',  # Thai
    'tr': 'tur',  # Turkish
    'ur': 'urd',  # Urdu
    'vi': 'vie',  # Vietnamese
    'zh': 'zho',  # Chinese
    'cs': 'ces',  # Czech
    'gl': 'glg',  # Galician
    'fi': 'fin',  # Finnish
    'is': 'isl',  # Icelandic
    'it': 'ita',  # Italian
    'ja': 'jpn',  # Japanese
    'ko': 'kor',  # Korean
    'pl': 'pol',  # Polish
    'pt': 'por',  # Portuguese
    'id': 'ind',  # Indonesian
    'sv': 'swe',  # Swedish
}

def get_langs_three_digits(languages):
    langs_three_digits = []
    for lang in languages:
        if lang in language_code_map:
            langs_three_digits.append(language_code_map[lang])
        else:
            print(f"Warning: Language '{lang}' not found in mapping.")
    return langs_three_digits


def get_distances(languages):
    langs_three_digits = get_langs_three_digits(languages)
    distance_matrices = defaultdict(list)
    distance_dfs = defaultdict(pd.DataFrame)
    for distance in l2v.DISTANCES:
        distance_matrices[distance] = l2v.distance(distance, langs_three_digits)

        distance_matrix = distance_matrices[distance]

        distance_dfs[distance] = pd.DataFrame(distance_matrix, index=languages, columns=languages)

    return distance_dfs


def invert_structure(original):
    transformed = defaultdict(lambda: defaultdict(dict))

    for lang_eval in original:
        for lang_train in original[lang_eval]:
            for data_frac in original[lang_eval][lang_train]:
                metrics = original[lang_eval][lang_train][data_frac]
                # Initialize nested dict if needed
                if data_frac not in transformed[lang_train][lang_eval]:
                    transformed[lang_train][lang_eval][data_frac] = {}
                # Copy all metrics
                transformed[lang_train][lang_eval][data_frac].update(metrics)

    return dict(transformed)

def plot_results(results, lang_train, metrics='all'):
    if lang_train not in results:
        print(f"No results for training language: {lang_train}")
        return

    # Collect all unique metrics from the data
    if metrics == 'all':
        metrics = ["accuracy", "precision_micro", "recall_micro", "f1_micro", "precision_macro", "recall_macro", "f1_macro"]
    elif isinstance(metrics, str):
        metrics = [metrics]  # Convert single metric to list

    all_metrics = sorted(metrics)
    n_metrics = len(all_metrics)

    # Determine subplot layout (2 columns)
    n_cols = 3
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5 * n_rows), sharex=True)
    axes = axes.flatten()  # Flatten in case of single row

    # Generate enough distinct colors
    lang_evals = sorted(results[lang_train].keys())
    colors = plt.get_cmap('tab20').colors
    color_map = {lang_eval: colors[i % len(colors)] for i, lang_eval in enumerate(lang_evals)}

    for i, metric in enumerate(all_metrics):
        ax = axes[i]

        for lang_eval in lang_evals:
            data_fractions = []
            values = []

            for data_frac in sorted(results[lang_train][lang_eval]):
                metrics = results[lang_train][lang_eval][data_frac]
                if metric in metrics and metrics[metric] is not None:
                    data_fractions.append(float(data_frac))
                    values.append(metrics[metric])

            if data_fractions:
                x_y = sorted(zip(data_fractions, values))
                x_sorted, y_sorted = zip(*x_y)
                ax.plot(x_sorted, y_sorted, 'o-', label=lang_eval, color=color_map[lang_eval], alpha=0.8)

        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} vs. Data Fraction (Source language='{lang_train}')")
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.legend(title="lang_eval")

    # Hide any unused subplots
    for j in range(len(all_metrics), len(axes)):
        fig.delaxes(axes[j])

    axes[-1].set_xlabel("Data Fraction Used for Training")
    plt.tight_layout()
    plt.show()


def compute_correlation(results_per_eval, distance_dfs, distance, metric='accuracy', data_fraction=1.0, visualize_heatmap=True):
    data_fraction = str(data_fraction)
    # 1. Extract accuracy and distances
    source_target_pairs = []
    accuracies = []
    distances = []

    for target_lang in results_per_eval:
        for source_lang in results_per_eval[target_lang]:
            acc = results_per_eval[target_lang][source_lang][data_fraction][metric]
            if acc is None:
                continue
            dist = distance_dfs[distance].loc[source_lang, target_lang]

            source_target_pairs.append((source_lang, target_lang))
            accuracies.append(acc)
            distances.append(dist)

    # 2. Correlation analysis
    correlation, pval = spearmanr(distances, accuracies)
    print(f"Spearman correlation for {distance} distance (data fraction = {data_fraction}): {correlation:.3f} (p = {pval:.3g})")

    # 4. Visualize metric heatmap if requested
    if visualize_heatmap:
        plot_heatmap(source_target_pairs, accuracies, metric=metric, data_fraction=data_fraction)

    return correlation, pval


def plot_heatmap(source_target_pairs, accuracies, metric='accuracy', data_fraction='1.0', languages=None):
    heatmap_data = pd.DataFrame(index=languages, columns=languages)
    for (src, tgt), acc in zip(source_target_pairs, accuracies):
        heatmap_data.loc[src, tgt] = acc
    heatmap_data = heatmap_data.astype(float)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, vmin=0, vmax=1)
    plt.title(f"{metric.capitalize()} of Transfer (source → target) (data fraction = {data_fraction})")
    plt.xlabel("Target Language")
    plt.ylabel("Source Language")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_distance_vs_accuracy_per_source(results_per_eval, languages, distance_dfs, distance, metric='accuracy', data_fraction=1.0):
    data_fraction = str(data_fraction)

    n_cols = 4
    n_rows = int(np.ceil(len(languages) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for i, source_lang in enumerate(languages):
        ax = axes[i]
        distances = []
        accuracies = []

        for target_lang in languages:
            if target_lang == source_lang:
                continue  # skip same-language
            try:
                acc = results_per_eval[target_lang][source_lang][data_fraction][metric]
                if acc is None:
                    continue
                dist = distance_dfs[distance].loc[source_lang, target_lang]
                distances.append(dist)
                accuracies.append(acc)
            except KeyError:
                continue

        if distances and accuracies:
            ax.scatter(distances, accuracies, alpha=0.7)
            corr, _ = spearmanr(distances, accuracies)
            ax.set_title(f"{source_lang} (ρ={corr:.2f})")
        else:
            ax.set_title(f"{source_lang} (no data)")

        ax.set_xlim(0, 1)
        ax.set_xlabel("Featural Distance")
        if i % n_cols == 0:
            ax.set_ylabel("Accuracy")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Accuracy vs. Featural Distance (data_fraction = {data_fraction})", fontsize=16)
    plt.ylim(0, 1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_accuracy_vs_data_fraction_per_source(results_per_train, fractions, metric='accuracy'):
    source_langs = list(results_per_train.keys())
    all_target_langs = sorted({tgt for src in results_per_train for tgt in results_per_train[src]})

    # Assign fixed colors to each target language
    cmap_colors = plt.get_cmap('tab20').colors
    color_map = {lang: cmap_colors[i % len(cmap_colors)] for i, lang in enumerate(all_target_langs)}

    n_sources = len(source_langs)
    n_cols = 3
    n_rows = int(np.ceil(n_sources / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    all_handles = {}

    for i, source_lang in enumerate(source_langs):
        ax = axes[i]
        for target_lang in results_per_train[source_lang]:
            present_fractions = []
            values = []

            for frac in fractions:
                try:
                    score = results_per_train[source_lang][target_lang][frac][metric]
                    if score is not None:
                        present_fractions.append(float(frac))
                        values.append(score)
                except KeyError:
                    continue

            if present_fractions:
                x_y = sorted(zip(present_fractions, values))
                x_sorted, y_sorted = zip(*x_y)
                color = color_map[target_lang]
                line, = ax.plot(x_sorted, y_sorted, label=target_lang, marker='o', color=color)
                all_handles[target_lang] = line  # store one handle per target

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(f"{source_lang}")
        ax.set_xlabel("Data Fraction")
        if i % n_cols == 0:
            ax.set_ylabel(metric.capitalize())
        ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    labels, handles = zip(*sorted(all_handles.items()))

    plt.tight_layout(rect=[0, 0.0, 1, 0.93])
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=min(len(labels), 8),title="Target Language")
    fig.suptitle(f"{metric.capitalize()} vs. Fine-Tuning Data Fraction (per source language)", fontsize=16)
    plt.show()

def plot_violin_source_transfer_performance(results_per_train, data_fraction=1.0, metric='accuracy'):
    data_fraction = str(data_fraction)  # in case keys are strings

    records = []

    for source_lang in results_per_train:
        for target_lang in results_per_train[source_lang]:
            if target_lang == source_lang:
                continue  # skip self-transfer if not needed
            try:
                score = results_per_train[source_lang][target_lang][data_fraction][metric]
                if score is not None:
                    records.append({
                        'Source': source_lang,
                        'Target': target_lang,
                        'Accuracy': score
                    })
            except KeyError:
                continue

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Plot violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Source', y='Accuracy', inner='box', palette='muted')
    plt.title(f"Distribution of Transfer Performance per Source Language (data fraction = {data_fraction})")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_language_rank_across_targets(results_per_train, data_fractions, metric='accuracy', distance_dfs=None, distance=None, language_code='en'):
    rank_records = []

    for frac in data_fractions:
        frac_key = str(frac)

        for target_lang in results_per_train:
            if target_lang == language_code:
                continue  # skip self-transfer

            source_scores = []
            for source_lang in results_per_train[target_lang]:
                if source_lang == target_lang:
                    continue  # optionally skip self-transfer
                try:
                    if metric == "distance":
                        score = distance_dfs[distance].loc[source_lang, target_lang]
                    else:
                        score = results_per_train[source_lang][target_lang][frac_key][metric]
                    if score is not None:
                        source_scores.append((source_lang, score))
                except KeyError:
                    continue
            # Rank sources by descending score, giving using max method to resolve ties
            source_scores_df = pd.DataFrame(source_scores, columns=['Source', 'Score'])
            source_scores_df["rank"] = source_scores_df['Score'].rank(ascending=(metric == 'distance'), method='average')
            perf_ranks = source_scores_df.set_index('Source')['rank'].to_dict()


            # Get English rank (if available)
            if language_code in perf_ranks:
                english_rank = perf_ranks[language_code]
                total_sources = len(perf_ranks)
                rank_records.append({
                    'Target': target_lang,
                    'Data Fraction': frac,
                    'Rank': english_rank,
                    'Total Sources': total_sources
                })
    return pd.DataFrame(rank_records)

def plot_ranks(ranks, dist_ranks, language_codes='en'):
    if isinstance(language_codes, str):
        language_codes = [language_codes]
    
    num_langs = len(language_codes)
    ncols = 3  # number of plot *pairs* per row
    nrows = math.ceil(num_langs / ncols)

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(10 * ncols, 6 * nrows),
                             gridspec_kw={'width_ratios': [4, 1] * ncols},
                             sharey='row')
    
    # Normalize axes shape
    if num_langs == 1:
        axes = [axes]
    elif nrows == 1:
        axes = [axes]
    else:
        axes = axes.reshape((nrows, ncols * 2))

    for idx, lang_code in enumerate(language_codes):
        row = idx // ncols
        col_offset = (idx % ncols) * 2
        ax_main = axes[row][col_offset]
        ax_dist = axes[row][col_offset + 1]

        ranks_df = ranks[lang_code]
        dist_ranks_df = dist_ranks[lang_code]

        sns.boxplot(data=ranks_df, x='Data Fraction', y='Rank', ax=ax_main)
        ax_main.set_title(f"Rank of {code2name[lang_code]} as Source Language Across Targets", fontsize=20)
        ax_main.set_ylabel("Rank (lower is better)", fontsize=16)
        ax_main.set_xlabel("Data Fraction", fontsize=16)

        sns.boxplot(data=dist_ranks_df, x='Data Fraction', y='Rank', color='lightgray', 
                    fliersize=0, width=0.4, ax=ax_dist)
        ax_dist.set_xlabel("")
        ax_dist.tick_params(labelsize=16)


        
        for ax in [ax_main, ax_dist]:
            ax.grid(True)
            ax.set(ylim=(1, dist_ranks_df.shape[0] + 1))

    # Remove unused axes if total subplot slots > needed
    total_axes = nrows * ncols * 2
    for j in range(2 * num_langs, total_axes):
        row = j // (2 * ncols)
        col = j % (2 * ncols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_radar_sources_for_targets(results_per_train, target_langs, data_fraction=1.0, metric='accuracy'):
    data_fraction = str(data_fraction)
    source_langs = list(results_per_train.keys())
    print(len(source_langs), "source languages found")

    # Assign colors
    colors = plt.get_cmap("tab10").colors  # supports up to 10 targets distinctly
    color_map = {lang: colors[i % len(colors)] for i, lang in enumerate(target_langs)}

    # Get consistent labels across all targets (union of available sources)
    labels = sorted({src for src in source_langs if src not in target_langs})  # exclude targets as sources
    num_vars = len(labels)

    # Compute angles for the radar plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    # Start plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for target_lang in target_langs:
        accuracies = []

        for source_lang in labels:
            try:
                score = results_per_train[source_lang][target_lang][data_fraction][metric]
                accuracies.append(score if score is not None else 0)
            except KeyError:
                accuracies.append(0)

        # Close the radar loop
        accuracies.append(accuracies[0])
        color = color_map[target_lang]

        # Plot for this target
        ax.plot(angles, accuracies, marker='o', label=target_lang, color=color, linewidth=2, alpha=0.8)
        ax.fill(angles, accuracies, alpha=0.1, color=color)

    # Ticks and layout
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric.capitalize()} Transfer from Sources to Targets (data fraction = {data_fraction})", fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05), title="Target Language")
    plt.tight_layout()
    plt.show()


def plot_eval_language(results, langs_eval, metric='accuracy', task='POS Tagging'):
    plt.rcParams['font.size'] = 16
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    langs_train = sorted(results[langs_eval[0]].keys())

    all_lines = []
    all_labels = []

    for i, lang_eval in enumerate(langs_eval):
        ax = axes[i]
        for lang_train in langs_train:
            data_fractions = ['0.25', '0.5', '0.75', '1.0']
            values = [results[lang_eval][lang_train][d_fraq][metric] for d_fraq in data_fractions]
            if data_fractions:
                x_y = sorted(zip(data_fractions, values))
                x_sorted, y_sorted = zip(*x_y)
                line, = ax.plot(x_sorted, y_sorted, 'o-', label=lang_train,
                                color=code2color[lang_train], alpha=0.8, linewidth=3, markersize=8)
                if lang_train not in all_labels:
                    all_lines.append(line)
                    all_labels.append(lang_train)

        ax.set_title(f"Target {code2name[lang_eval]}")
        ax.set_xlabel("Data Fraction Used for Training")
        ax.grid(True)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel(metric.capitalize())

    fig.legend(all_lines, all_labels, title="Source lang.",
                bbox_to_anchor=(0.5, -0.1), loc='lower center',
                ncol=min(len(all_labels), 11), fontsize='small', frameon=False)

    plt.suptitle(f"{task} Cross-Lingual Transfer", fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
    plt.show()


def plot_accuracy_vs_data_fraction_per_eval(results_per_eval, fractions, metric='accuracy', task="", langs_eval=None):
    if langs_eval is None:
        langs_eval = sorted(results_per_eval.keys())
    langs_train = sorted(results_per_eval[langs_eval[0]].keys())
    n_sources = len(langs_train)
    n_cols = 3
    n_rows = int(np.ceil(n_sources / n_cols))

    plt.rcParams['font.size'] = 16
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharey=True)
    axes = axes.flatten()

    all_handles = {}

    for i, lang_eval in enumerate(langs_eval):
        ax = axes[i]
        for lang_train in results_per_eval[lang_eval]:
            present_fractions = []
            values = []

            for frac in fractions:
                score = results_per_eval[lang_eval][lang_train][frac][metric]
                if score is not None:
                    present_fractions.append(float(frac))
                    values.append(score)

            if present_fractions:
                x_y = sorted(zip(present_fractions, values))
                x_sorted, y_sorted = zip(*x_y)
                line, = ax.plot(x_sorted, y_sorted, label=lang_train, marker='o', color=code2color[lang_train])
                all_handles[lang_train] = line

        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0.25", "0.5", "0.75", "1.0"])
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.set_title(f"Target {code2name[lang_eval]}")
        ax.set_xlabel("Data Fraction Used for Training")
        if i % n_cols == 0:
            ax.set_ylabel(metric.capitalize())

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    labels, handles = zip(*sorted(all_handles.items()))

    plt.tight_layout(rect=[0, -0.3, 1, 0.96])
    plt.legend(handles, labels, title="Source lang.",
                bbox_to_anchor=(-0.7, -0.7), loc='lower center',
                ncol=min(len(labels), 11), fontsize='small', frameon=False)
    if task != "":
        task = f"{task}: "
    fig.suptitle(f"{task}{metric.capitalize()} vs. Fine-Tuning Data Fraction", fontsize=20)
    plt.show()