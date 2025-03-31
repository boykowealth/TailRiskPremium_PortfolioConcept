import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("C:/Users/Brayden Boyko/Downloads/322 Data CSV.csv")

df = df.melt(id_vars=['Year'], value_vars=['Revenue', 'EBIT', 'NOPAT'], 
             var_name='name', value_name='value')

df = df.dropna()

fig = px.line(df, 
              x='Year', 
              y='value', 
              color='name', 
              line_group='name', 
              animation_frame='Year',  # Animation on 'Year'
              title="Performance Metrics Over Time",
              labels={"Year": "Years", "value": "Value", "name": "Line Item"})

fig.update_layout(
    plot_bgcolor='#222',
    paper_bgcolor='#222',
    font=dict(color='white'),
    xaxis=dict(title=dict(text='Years', font=dict(color='white')), tickfont=dict(color='white')),
    yaxis=dict(title=dict(text='', font=dict(color='white')), tickfont=dict(color='white')),
    legend=dict(bgcolor='#222', font=dict(color='white'))
)

fig.update_traces(line=dict(width=1.2)) 
fig.update_yaxes(tickformat="$") 

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=500, redraw=True),
                                      fromcurrent=True)]),
                dict(label='Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')])
            ],
            x=1, xanchor='right',
            y=0, yanchor='bottom',
            font=dict(color='white')
        )
    ]
)

fig.update_layout(
    sliders=[
        dict(
            active=0,
            currentvalue=dict(prefix="Year: ", font=dict(color='white')),
            pad=dict(t=50),
            tickfont=dict(color='white'),
        )
    ]
)

# Show plot
fig.show()