import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


def visualize_contact_map(map, zmax):
    fig = px.imshow(map.squeeze(),zmax=zmax,color_continuous_scale="darkmint", width=500)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig