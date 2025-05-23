"""Wrapper for the bar plot from the ``shap`` package.

Note:
    Code and implementation was taken and adapted from the [SHAP package](https://github.com/shap/shap)
    which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).
"""

import matplotlib.pyplot as plt
import numpy as np

from ..interaction_values import InteractionValues, aggregate_interaction_values
from ._config import BLUE, RED
from .utils import abbreviate_feature_names, format_labels, format_value

__all__ = ["bar_plot"]


def _bar(
    values: np.ndarray,
    feature_names: np.ndarray,
    max_display: int | None = 10,
    ax: plt.Axes | None = None,
    sd_values: np.ndarray | None = None,
    aggregate_rest: bool = True,  # NEW OR MODIFIED PARAMETER
) -> plt.Axes:
    """
    Create a bar plot of a set of SHAP values, optionally with error bars.
    Adapted from the SHAP package (MIT license):
    https://github.com/shap/shap/blob/master/shap/plots/_bar.py

    Args:
        values (np.ndarray): Explanation values to plot, shape (n_groups, n_features).
        feature_names (np.ndarray): Feature/interaction labels, length == n_features.
        max_display (int | None): Max number of features to display. Defaults to 10.
        ax (plt.Axes | None): A Matplotlib Axes to draw on. If None, we create a new one.
        sd_values (np.ndarray | None): Standard deviations for error bars, matching
            (n_groups, n_features). If None, no error bars.
        aggregate_rest (bool): If True, aggregates all features beyond `max_display` into a single bar
            labeled "Sum of X other features". If False, we simply skip them. Defaults to True.

    Returns:
        plt.Axes: The axis on which the bar plot is drawn.
    """
    num_groups, num_features = values.shape
    if max_display is None:
        max_display = num_features
    max_display = min(max_display, num_features)
    num_cut = max(num_features - max_display, 0)

    # sort features by descending mean
    # feature_order = np.argsort(np.mean(values, axis=0))[::-1]  # TODO integrate properly
    feature_order = np.array([0, 2, 1])

    # if more features than max_display, aggregate "other" features
    if aggregate_rest and num_cut > 0:
        cut_values = values[:, feature_order[max_display:]]
        sum_of_remaining = np.sum(cut_values, axis=None)
        index_of_last = feature_order[max_display]
        values[:, index_of_last] = sum_of_remaining

        # handle sums of SDs (simple direct sum; adapt to sum-in-quadrature if needed)
        if sd_values is not None:
            cut_sds = sd_values[:, feature_order[max_display:]]
            sum_of_sds = np.sum(cut_sds, axis=None)
            sd_values[:, index_of_last] = sum_of_sds

        max_display += 1

    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    yticklabels = [feature_names[i] for i in feature_inds]

    if aggregate_rest and num_cut > 0:
        yticklabels[-1] = f"Sum of {num_cut} other features"

    # create or reuse the axes
    if ax is None:
        ax = plt.gca()
        fig = plt.gcf()
        row_height = 0.5
        if len(feature_names) > 0:
            max_label_len = max(len(fn) for fn in feature_names)
        else:
            max_label_len = 1
        fig.set_size_inches(
            8 + 0.3 * max_label_len,
            max_display * row_height * np.sqrt(num_groups) + 1.5,
        )

    # add a vertical line at x=0 if negative values
    negative_values_present = np.any(values[:, feature_inds] < 0)
    if negative_values_present:
        ax.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    # bar widths
    total_width = 0.7
    bar_width = total_width / num_groups

    # patterns for each group if desired
    patterns = (None, "\\\\", "++", "xx", "////", "*", "o", "O", ".", "-")

    # plot each group
    for i in range(num_groups):
        ypos_offset = -((i - num_groups / 2) * bar_width + bar_width / 2)

        x_errors = sd_values[i, feature_inds] if (sd_values is not None) else None

        ax.barh(
            y_pos + ypos_offset,
            values[i, feature_inds],
            bar_width,
            xerr=x_errors,
            align="center",
            color=[
                BLUE.hex if values[i, feature_inds[j]] <= 0 else RED.hex
                for j in range(len(y_pos))
            ],
            hatch=patterns[i] if i < len(patterns) else None,
            edgecolor=(1, 1, 1, 0.8),
            label=f"Group {i + 1}",
            error_kw={"elinewidth": 1, "capsize": 3},
        )

    formatted_labels = []
    for t in yticklabels:
        suffix = t.split("=")[-1]  # drop any leading "foo="
        if " x " in suffix:  # pattern:  A x B  (x surrounded by spaces)
            before, after = suffix.split(" x ", 1)
            formatted_labels.append(f"{before} x\n{after}")  # keep x on first line
        else:
            formatted_labels.append(suffix)
    # --------------------------------------------------
    # y ticks
    #ax.set_yticks(
    #    list(y_pos) + list(y_pos + 1e-8),
    #    # formatted_labels, # yticklabels + [t.split("=")[-1] for t in yticklabels],
    #    # yticklabels + [t.split("=")[-1] for t in yticklabels],
    #    formatted_labels, #  + [t.split("=")[-1] for t in formatted_labels],
    #    fontsize=18,
    #)
    ax.set_yticks(y_pos, formatted_labels, fontsize=22)

    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width

    # label each bar with numeric text
    for i in range(num_groups):
        ypos_offset = -((i - num_groups / 2) * bar_width + bar_width / 2)
        for j in range(len(y_pos)):
            ind = feature_inds[j]
            val = values[i, ind]
            text_x = val + (5 / 72) * bbox_to_xscale if val >= 0 else val - (5 / 72) * bbox_to_xscale
            ha = "left" if val >= 0 else "right"
            color = RED.hex if val >= 0 else BLUE.hex

            ax.text(
                text_x,
                float(y_pos[j] + ypos_offset),
                format_value(val, "%+0.02f"),
                ha=ha,
                va="center",
                color=color,
                fontsize=16,
            )

    # horizontal lines
    for i in range(max_display):
        ax.axhline(i + 1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)

    # style
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if negative_values_present:
        ax.spines["left"].set_visible(False)
    ax.tick_params("x", labelsize=15)

    # expand x-limits slightly
    xmin, xmax = ax.get_xlim()
    x_buffer = 0.05 * (xmax - xmin)
    if negative_values_present:
        ax.set_xlim(xmin - x_buffer, xmax + x_buffer)
    else:
        ax.set_xlim(xmin, xmax + x_buffer)

    ax.set_xlabel("Attribution", fontsize=16)

    if num_groups > 1:
        ax.legend(fontsize=16, loc="lower right")

    # color y-tick labels
    #tick_labels = ax.yaxis.get_majorticklabels()
    #for i in range(max_display):
    #    tick_labels[i].set_color("#999999")

    return ax


def bar_plot(
    list_of_interaction_values: list[InteractionValues] | list[list[InteractionValues]],
    feature_names: np.ndarray | None = None,
    show: bool = False,
    abbreviate: bool = True,
    max_display: int | None = 10,
    global_plot: bool = True,
    plot_base_value: bool = False,
    ax: plt.Axes | None = None,
    sd_values: np.ndarray | None = None,
    aggregate_rest: bool = False,  # NEW OR MODIFIED PARAMETER
) -> plt.Axes | None:
    """
    Draws interaction values as a SHAP bar plot, optionally with error bars and optionally
    handling a list of lists of InteractionValues.

    If you pass a list of lists:
        1) For each inner list, we call the unmodified 'aggregate_interaction_values(...)'
           to merge them (mean by default).
        2) We also compute a per-feature standard deviation across that inner list to
           display as error bars in the final plot.

    Args:
        list_of_interaction_values: Either
            - A list of InteractionValues, or
            - A list of lists of InteractionValues (multiple sublists).
        feature_names: Names for each feature. If None, uses F0..F(n_players-1).
        show: If True, calls plt.show() before returning. Defaults to False.
        abbreviate: Whether to abbreviate feature names. Defaults to True.
        max_display: Max number of features to display. Defaults to 10.
        global_plot: If True and multiple InteractionValues are passed, aggregates them
            (with abs(...) then mean). If False, each IV is plotted separately.
        plot_base_value: Whether to include the base value (interaction=()) in the plot.
            Defaults to False.
        ax: A Matplotlib Axes to draw on. If None, a new figure/axes is created.
        sd_values: (n_groups, n_features) array of standard deviations for error bars,
            if you already have them. Usually not needed if passing a list of lists,
            because we auto-compute SD from each sublist.
        aggregate_rest (bool): If True, merges all features beyond `max_display` into
            a single "rest" bar. If False, omit them entirely.

    Returns:
        plt.Axes | None: The Matplotlib Axes if show=False, otherwise None after plt.show().
    """
    # OLD COMMENT:
    # "for sublist in list_of_interaction_values" => aggregated_iv = aggregate_interaction_values(sublist)

    # NEW:
    # We interpret the outer dimension as #configurations, and the inner dimension as #samples.
    # We want the final result to be of size n_samples, i.e. for each sample index i we aggregate
    # across all configurations.

    auto_computed_sd_values = None
    outer_list = list_of_interaction_values

    if outer_list and isinstance(outer_list[0], list):
        # The outer list length = n_config, each sublist is of length n_samples
        n_config = len(outer_list)
        n_samples = len(outer_list[0])

        # Check all sublists have the same length
        for conf_sublist in outer_list:
            if len(conf_sublist) != n_samples:
                raise ValueError("All inner sublists must have the same number of samples.")

        aggregated_samples = []
        all_std_arrays = []

        # For each sample index i, gather that sample from each configuration
        for i in range(n_samples):
            # collect the i-th InteractionValues from each config
            sample_across_configs = [outer_list[c][i] for c in range(n_config)]

            # aggregate across configs
            mean_iv = aggregate_interaction_values(sample_across_configs, aggregation="abs_mean")
            aggregated_samples.append(mean_iv)

            # compute stdev across configs for each interaction
            keys = sorted(mean_iv.interaction_lookup.keys())
            std_array = np.zeros(len(keys), dtype=float)
            for idx, key in enumerate(keys):
                vals = [iv[key] for iv in sample_across_configs]
                std_array[idx] = np.std(vals, ddof=1)  # sample std dev
            all_std_arrays.append(std_array)

        # Now we have a list of n_samples aggregated IVs
        list_of_interaction_values = aggregated_samples

        # shape => (n_samples, n_features)
        auto_computed_sd_values = np.vstack(all_std_arrays)

    # -------------------------------------------------------------------------
    # 2) Now handle normal logic with either a list of IVs or the newly aggregated one
    # -------------------------------------------------------------------------
    n_players = list_of_interaction_values[0].n_players

    if feature_names is not None:
        if abbreviate:
            feature_names = abbreviate_feature_names(feature_names)
        feature_mapping = {i: feature_names[i] for i in range(n_players)}
    else:
        feature_mapping = {i: f"F{i}" for i in range(n_players)}

    # If user also provided sd_values, we override auto_computed_sd_values
    if sd_values is not None:
        auto_computed_sd_values = sd_values

    # Decide on "global" vs separate
    if global_plot and len(list_of_interaction_values) > 1:
        # sum up absolute values, then average them
        abs_list = [abs(iv) for iv in list_of_interaction_values]
        global_values = aggregate_interaction_values(abs_list, "mean")

        values = global_values.values[np.newaxis, :]  # shape => (1, n_features)
        interaction_list = list(global_values.interaction_lookup.keys())

        # if we have auto_computed_sd_values, reduce it similarly or do something custom
        # but typically it doesn't make sense to "average" the SDs again. We'll skip that here.
        if auto_computed_sd_values is not None and auto_computed_sd_values.ndim == 1:
            auto_computed_sd_values = auto_computed_sd_values[np.newaxis, :]

    else:
        # gather all interactions
        all_interactions = set()
        for iv in list_of_interaction_values:
            all_interactions.update(iv.interaction_lookup.keys())
        all_interactions = sorted(all_interactions)
        interaction_list = list(all_interactions)

        values = np.zeros((len(list_of_interaction_values), len(all_interactions)))
        for j, interaction in enumerate(all_interactions):
            for i, iv in enumerate(list_of_interaction_values):
                values[i, j] = iv[interaction]

        # if we have auto_computed_sd_values, ensure shape matches
        if auto_computed_sd_values is not None:
            if auto_computed_sd_values.shape != values.shape:
                raise ValueError(
                    f"Computed or provided SD shape {auto_computed_sd_values.shape} "
                    f"does not match bar values shape {values.shape}!"
                )

    # optionally remove base value
    if not plot_base_value:
        values = values[:, 1:]
        interaction_list = interaction_list[1:]
        if auto_computed_sd_values is not None:
            auto_computed_sd_values = auto_computed_sd_values[:, 1:]

    # format final labels
    labels = [format_labels(feature_mapping, inter) for inter in interaction_list]

    # Plot
    final_ax = _bar(
        values=values,
        feature_names=np.array(labels, dtype=object),
        max_display=max_display,
        ax=ax,
        sd_values=auto_computed_sd_values,
        aggregate_rest=aggregate_rest,
    )

    if show:
        plt.show()
        return None
    else:
        return final_ax
