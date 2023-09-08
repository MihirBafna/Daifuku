import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import ipywidgets as widgets


def visualize_contact_map(map, zmax, title=""):
    fig = px.imshow(map.squeeze(),zmax=zmax,color_continuous_scale="teal", width=500)

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        title=title,
    )
    return fig

def visualize_multiple_jupyter(maps):
    figurewidgets= []
    for i in range(len(maps)):
        figurewidgets.append(go.FigureWidget(maps[i]))

    return widgets.HBox(figurewidgets)