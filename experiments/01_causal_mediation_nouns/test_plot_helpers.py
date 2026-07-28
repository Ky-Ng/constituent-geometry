from plot_helpers import plot_heatmap

Z = [
    [0.5, 0, 0.5],
    [1.0, 0, 0]
]

Z_text = [
    ["1a", "1笔", "1c"],
    ["2a", "2b", "2c"]
]

x_labels = ["诶", "笔", "是"]

y_labels = ["1", "2"]

x_axis_label = "letters"
y_axis_label = "numbers"

title = "Test of Numbers vs. Letters"

out_path = "./temp"

plot_heatmap(
    Z=Z,
    Z_text=Z_text,
    x_labels=x_labels,
    y_labels=y_labels,
    x_axis_label=x_axis_label,
    y_axis_label=y_axis_label,
    title=title,
    out_path=out_path   
)