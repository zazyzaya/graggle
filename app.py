import plotly.graph_objects as go
import networkx as nx
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import random 
import copy
import dash_table

from dash.exceptions import PreventUpdate
from simple_search import query
from load_data import g, df, graph_df
from dash.dependencies import Output, Input, State

####### DATA PROCESSING ########
build_titles = lambda ids : [
    df['title'][i] + '<br>' 
    + format_journal(df['journal'][i])
    + format_authors(df['authors'][i]) + str(df['publish_time'][i]) 
    for i in ids
]

build_link = lambda records : [
    {
        'link': format_link(df['url'][r['index']]),
        'index': r['index'],
        'title': r['title']
    } for r in records
]

def format_link(l):
    if str(l) == 'nan':
        return 'Not Found'
    else:
        if ';' in l:
            l = l.split(';')[0]

        return '[&#8599;]' + '(' + l + ')'

def format_journal(s):
    if str(s) == 'nan':
        return ''
    
    return s + ' &#8212; ' 

def format_authors(s):
    if str(s) == 'nan':
        return ''
    
    spl = s.split(';')
    if len(spl) > 3:
        return spl[0] + ' et al., '
    else:
        return s + ' '

def get_all_g(n=None, hops=None, min_edge=None, max_edges=None):
    Xn = graph_df['X0']
    Yn = graph_df['X1']
    ids = list(graph_df.index)
    titles = build_titles(ids)
    labels=graph_df['y']
    
    return Xn, Yn, [], titles, labels, ids

def get_data_v(n=None, hops=3, min_edge=10, max_edges=10):
    ''' 
    Places nodes based on their node2vec embedding position
    after T-SNE dim redux
    '''
    # Don't display anything for nodes not in data
    if n != None and n not in graph_df.index:
        return [], [], [], [], [], []
    
    if n == None:
        n = random.randint(0, len(graph_df))
        
    # Build subgraph from n
    explored = set()
    to_explore = [n]
    nodes = set()
    edges = []
    for i in range(hops):
        frontier = []
        for node in to_explore:
            if node in explored:
                continue
            
            nodes.add(node)
            for idx in np.where(g[node].data > min_edge)[0][:max_edges]:
                nodes.add(g[node].indices[idx])
                edges.append((node, g[node].indices[idx], g[node].data[idx]))
                
                frontier.append(g[node].indices[idx])
            explored.add(node)

        to_explore = frontier
    
    ids = list(nodes)    
    Xn = graph_df['X0'][ids]
    Yn = graph_df['X1'][ids]
    
    # Kind of a hack, but emphasize the clicked node by plotting it as
    # a single large point
    lines = [
        go.Scatter(
            x=[graph_df['X0'][n]],
            y=[graph_df['X1'][n]],
            mode='markers',
            marker=dict(
                symbol='star',
                size=20,
                color=graph_df['y'][n],
                colorscale='Viridis'
            ),
            text=[None],
            hoverinfo='text',
            customdata=[n]
        )
    ]
    
    for e in edges:
        lines.append(
            go.Scatter(
                x=graph_df['X0'][list(e[:2])],
                y=graph_df['X1'][list(e[:2])],
                mode='lines',
                line=dict(
                    color='rgb(125,125,125)', 
                    width=int((e[2]/min_edge)/2)+1
                ),
                opacity=0.25
            )
        )
        
    titles=build_titles(ids)
    labels=graph_df['y'][ids]
    
    return Xn, Yn, lines, titles, labels, ids

def get_data(n=None, hops=3, min_edge=25, max_edges=10):
    if n != None and n not in df.index:
        return [], [], [], [], [], []
    
    if n == None:
        n = random.randint(0, len(df))
        
    graph = nx.Graph()
    graph.add_node(n)
    
    # Build subgraph from n
    explored = set()
    to_explore = [n]
    for i in range(hops):
        frontier = []
        for node in to_explore:
            if node in explored:
                continue
            
            for idx in np.where(g[node].data > min_edge)[0][:max_edges]:
                graph.add_edge(
                    node, 
                    g[node].indices[idx], 
                    weight=g[node].data[idx]
                )
                
                frontier.append(g[node].indices[idx])
            explored.add(node)
            
        to_explore = frontier
     
    # Convert to 2d       
    G_layout = nx.spring_layout(graph, dim=2)
    
    # Converting layout into X,Y coordinates for nodes and edges
    Xn = [G_layout[k][0] for k in G_layout.keys()]
    Yn = [G_layout[k][1] for k in G_layout.keys()]
    ids = [k for k in G_layout.keys()]
    
    titles = build_titles(list(G_layout.keys()))
    
    # There are ~100 unclassified nodes, but I assume this won't be
    # an issue
    lables = graph_df['y'][list(G_layout.keys())]

    # Kind of a hack, but emphasize the clicked node by plotting it as
    # a single large point
    edges = [
        go.Scatter(
            x=[G_layout[n][0]],
            y=[G_layout[n][1]],
            mode='markers',
            marker=dict(
                symbol='star',
                size=20,
                color=graph_df['y'][n],
                colorscale='Viridis'
            ),
            text=[None],
            hoverinfo='text',
            customdata=[n]
        )
    ]
    for e in graph.edges(data=True):
        # I tried making shapes, but apperently we need to make a million
        # tiny scatter plots
        edges.append(
            go.Scatter(
                x=[G_layout[e[0]][0],G_layout[e[1]][0]],
                y=[G_layout[e[0]][1],G_layout[e[1]][1]],
                mode='lines',
                line=dict(
                    color='rgb(125,125,125)', 
                    width=int((e[2]['weight']/min_edge)/2)+1
                )
            )
        )
    
    return Xn, Yn, edges, titles, lables, ids

def build_data(Xn, Yn, titles, labels, ids):
    nodes = go.Scatter(
        x=Xn,
        y=Yn,
        mode='markers',
        name='nodes',
        marker=dict(symbol='circle',
                    size=10,
                    color=labels,
                    colorscale='Viridis',
                    line=dict(color='rgb(50,50,50)', width=1)
                    ),
        text=titles,
        opacity=0.8,
        hoverinfo='text',
        customdata=ids
    )
    
    return [nodes]


######## WEB LAYOUT ########
l = go.Layout(
    title={
        'text': 'Related Papers',
        'x': 0.5,
        'xanchor': 'center'
    },
    showlegend=False,
    autosize=True,
    margin=dict(
        t=25,
        b=0,
        l=0,
        r=0
    ),
    hovermode='closest',
    clickmode='event',
)

#Xn, Yn, edges, titles, labels, ids = get_all_g()
data = [] 
last_id = None
fig = go.Figure(data=data, layout=l)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

num_papers = '{:,}'.format(len(graph_df))
num_papers = num_papers.split(',')[0] + ' thousand'

app = dash.Dash(
	__name__, 
	url_base_pathname='/graggle/'
)

with open('description.md', 'r') as f:
    desc = f.read()

# Add in google analytics info
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-168853683-1"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());

        gtag('config', 'UA-168853683-1');
        </script>

        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
app.title = 'Graggle: A Graph-based Search Engine for COVID-19'
app.layout = html.Div([     
    # Header
    html.Div([
        html.H1(
            'Graggle', 
            style={
                'margin-bottom': '0px'
            }
        ),
        html.H4(
            'A graph-based search engine for COVID-19 papers.',
            style = {
                'margin-top': '2px'
            }
        ),

        # Affiliations
        html.Div([
            html.A([
                html.Img(
                    src=app.get_asset_url('graph_lab_logo_ben.png'),
                    style={
                        'height': '40px',
                        'display': 'inline-block',
                        'vertical-align': 'top',
                        'margin-right': '10px'
                    }
                )],
                href='https://www2.seas.gwu.edu/~howie/glab.html',
                target='_blank'
            ),
            html.A([
                html.Img(
                    src=app.get_asset_url('gw-logo.png'),
                    style={
                        'height': '40px',
                        'display': 'inline-block',
                        'vertical-align': 'top'
                    }
                )],
                href='https://www.seas.gwu.edu/',
                target='_blank'
            )],
        style={
            'float': 'right',
            'padding-right': '17px',
            'vertical-align': 'top',
            'margin-top': '12px',
            'top': 5,
            'right': 0,
            'position': 'absolute'
        })
        ],
        
        style={
            'top': '0px',
            'width': '100%',
            'height': '75px',
            'color': 'white',
            'text-align': 'center',
            'background': "linear-gradient(90deg, rgba(3,60,90,1) 0%, rgba(3,44,65,1) 100%)",
            'padding-top': '10px',
            'margin-top': '-20px',
            'padding-bottom': '15px'
        }
    ),
    
    # Search bar
    html.Div([
        html.Div([
            dcc.Input(
                id='paper-search',
                type='text',
                placeholder='Keywords, e.g., virus, infection, vaccine, influenza, sequence, etc.',
                value='',
                debounce=True,
                style={
                    'display': 'inline-block',
                    'width': '60%',
                    'height': '30px',
                    'border-radius': '10px',
                    'margin-right': '5px',
                    'font-size': '15px'
                }
            ),
            html.Button(
                'Graggle Search', 
                id='search-button',
                style={
                    'display': 'inline-block',
                    'border-radius': '4px',
                    'height': '30px',
                }
            ),
            ],
            style={
                'width': '100%',
                'position': 'relative',
                'top': '50%',
                '-ms-transform': 'translateY(-50%)',
                'transform': 'translateY(-50%)'
            }
        ),
        ],
        style={
            'background': "linear-gradient(90deg, rgba(170,152,104,1) 0%, rgba(140,126,88,1) 100%)",
            'width': '100%',
            'height': '60px',
            'text-align': 'center',
        }
    ),
    
    # Brief description
    html.Div([
            html.P([
                'Graggle is an experimental search engine that uses a novel graph-based datastructure to identify important links between papers. To use it, simply search for papers as you normally would in the search bar. The resulting graph displayed is the k-hop neighbors of the paper from your search.' +
                ' Try it now, and query our database of over %s papers! ' % num_papers,
                html.A(
                    'Read more', href='#desc'
                )
            ],
            style={
                'width': '100%'
            }),
        ],
        style={
            'width': '90%',
            'margin-left': '50px',
            'margin-right': '250px',
            'margin-top': '5px'
        }
    ),

    # Middle
    html.Div([
    
        # Tools 
        html.Div([
            dcc.Checklist(
                    id='disp-method',
                    options=[
                        {
                            'label': 'Use t-SNE coords',
                            'value': 'yes'
                        }
                    ],
                    value=[]
            ),
            html.Div([
                dcc.Input(
                    id='node-id',
                    type='number',
                    value=random.randint(0,len(df)),
                )
            ], 
            style={'display': 'none'}  
            ), 
            

            #dcc.Slider(
            #    id='num-neighbors',
            #    min=0,
            #    max=100,
            #    step=1,
            #    value=10,
            #    marks={
            #        k : str(k) for k in range(10,100,10)
            #    },
            #),

            html.P(['K-Hop neighbors:']),
            dcc.Slider(
                id='num-hops',
                min=0,
                max=4,
                step=1,
                value=2,
                marks={
                    k : str(k) for k in range(1,4)
                }
            ),
            html.Div([
                html.P(['Minimum edge weight:']),
                dcc.Input(
                    id='e-weight',
                    min=0,
                    max=1000,
                    type='number',
                    value=10
                )
            ]),
            html.P(['Search results:']),
            html.Div([
                dash_table.DataTable(
                    id='search-results',
                    columns=[
                        {'name': 'Paper ID', 'id': 'index'},
                        {'name': 'Title', 'id': 'title', 'presentation': 'markdown'},
                        {'name': 'Link', 'id': 'link', 'presentation': 'markdown'}
                    ],
                    data=[],
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_table={
                        'maxHeight': '30vh',
                        'overflowY': 'scroll'
                    },
                    style_cell={
                        'textAlign': 'left'
                    },
                    style_cell_conditional=[{
                    'if': {'column_id': 'index'},
                    'display': 'none' 
                    }],
                    row_selectable='single'
                )],
            )
            ],
            style={ 
                'margin-left': 50, 
                'float': 'left',
                'width': '28%',
                'margin-top': 15
            }         
        ),
        
        # Graph
        html.Div([
            dcc.Graph(
                id='live-graph',
                figure=fig, 
                config={'scrollZoom': True, 'staticPlot': False}, 
                animate=True,
                animation_options={'frame': {'redraw': True}},
                style={
                    'width': '95%',
                    'height': 'calc(70vh - 50px)'
                }
            ),
            html.Div([
                html.A(
                        html.Button(
                            'View Selected Paper',
                            style={
                                'width': '85%', 
                                'height': '50px',
                                'font-size': '15px',
                                'color': '#FFF',
                                'background-color': 'rgba(3,60,90,1)',
                                'border-radius': '10px',
                                'vertical-align': 'top'
                            },
                            id='visit-button'
                        ),
                        href='',
                        target='_blank',
                        id='visit',
                ),

                html.Button(
                    "Undo",
                    id='undo',
                    style={
                        'width': '12%',
                        'color': '#FFF',
                        'background-color': 'rgba(3,60,90,1)',
                        'height': '50px',
                        'font-size': '15px',
                        'margin-left': '5px',
                        'border-radius': '10px',
                        'vertical-align': 'top'
                    }
                )],
                style={
                    'width': '95%',
                    'margin-top': '5px'
                }
            )
            ],
            style={
                'width': '65%',
                'float': 'right',
                'height': '80vh',
                'text-align': 'center'
            }
        )],
        style={
            'width': '100%',
            'float': 'left'
        }    
    ),
    
    # About
    html.Div([
            html.H3(['About']),
            dcc.Markdown([desc])    
        ],
        style={
            'margin-left': '50px',
            'margin-right': '50px',
            'float': 'left'
        },
        id='desc'
             
    ),
    
    # Spacer
    html.Div([], style={'flex': 1}),
    
    # Footer
    html.Div([
        # Social links
        html.Div([
            html.A([
                html.Div(
                    ['Suggestions'],
                    style={
                        'line-height': '25px',
                        'vertical-align': 'bottom',
                        'text-align': 'center',
                        'display': 'inline-block',
                        'font-size': '20px',
                        'color': 'white',
                        'padding-right': '12px',
                        'height': '40px'
                    }
                )],
                href='mailto:iking5@gwu.edu?cc=howie@gwu.edu&subject=CORD-19%20graph',
                target='_top',
                style={
                    'text-decoration': 'none',
                }
            ),
            html.A([
                html.Img(
                    src=app.get_asset_url('github-icon.png'),
                    style={
                        'display': 'inline-block',
                        'width': '50px',
                        'vertical-align': 'bottom'
                    }
                )],
                href='https://github.com/zazyzaya/graggle',       
                target='_blank'
            )    
            ],
            style={
                'float': 'left',
                'margin-top': '12px',
                'margin-left': '50px',
                'display': 'inline-block',
                'vertical-align': 'bottom'
            }
        ),
        ],
             
        style={
            'bottom': 0,
            'padding-bottom': 20,
            'padding-top': 5,
            'width': '100%',
            'height': '50px',
            'background-color': '#033C5A',
            'float': 'left'
        }
    )],
    style={
        'font-family': 'sans-serif',
        'color': 'rgba(3,60,90,1)'
    }
)


######## CALLBACKS ########

@app.callback(
    Output('live-graph', 'figure'),
    [   
        Input('disp-method', 'value'),
        Input('node-id', 'value'),
        Input('num-hops', 'value'),
        Input('e-weight', 'value'),
    ],
    [State('live-graph', 'figure')]
)
def update_graph(disp, nid, nhops, ew, current):
    if nid == None:
        return current

    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id']
    
    if len(disp) == 0:
        disp = get_data
    else:
        disp = get_data_v
    
    params = {
        'n': nid,
        'hops': nhops,
        'min_edge': ew
    }
    
    # Clean in case boxes are left empty it just defaults 
    if params['min_edge'] == 0:
        params.pop('min_edge')
    
    for k,v in params.copy().items():
        if v == None:
            params.pop(k)
    
    Xn, Yn, edges, titles, labels, ids = disp(**params)
    
    nodes = build_data(Xn, Yn, titles, labels, ids)

    return {
        'data': edges[1:]+nodes+edges[0:1],
        'layout': l,
    } 
 
@app.callback(
    [Output('search-results', 'data'),
     Output('search-results', 'selected_rows')],
    [Input('search-button', 'n_clicks'),
     Input('paper-search', 'n_submit')],
    [State('paper-search', 'value')]
)   
def search_papers(n, nn, text):
    if text.strip() == '':
        return [], []
    
    ids = query(text)
    data = df['title'][ids].reset_index()[['index', 'title']].to_dict('records')
    
    return build_link(data), [0]
        

@app.callback(
    Output('node-id', 'value'), 
    [Input('search-results', 'selected_rows'),
     Input('live-graph', 'clickData'),
     Input('undo', 'n_clicks')],
    [State('search-results', 'data'),
     State('node-id', 'value')]
)    
def select_row(idx, cd, n, rows, old):
    global last_id

    if (rows==[] or idx==[]) and cd == None:
        raise PreventUpdate
    
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id']
    
    if 'search-results' in trigger:
        newId = rows[idx[0]]['index']
    
    elif 'live-graph' in trigger:
        newId = cd['points'][0]['customdata']

    elif 'undo' in trigger:
        newId = last_id 

    last_id = old
    return newId

@app.callback(
    [Output('visit', 'href'),
     Output('visit-button', 'children')],
    [Input('node-id', 'value')]
)
def view_paper(nid):
    url = df['url'][nid].split(';')[0]
    title = df['title'][nid]

    while len(title) > 80:
        title = ' '.join(title.split(' ')[:-1]) + '...'
    
    title = 'Access the paper "' + title + '"'

    if str(url) == 'nan':
        return 'http://graphlab.seas.gwu.edu/404.html', title
    else:
        return url, title 
        
    

######## START EVERYTHING ########    
if __name__ == '__main__':
	app.run_server(debug=True, use_reloader=True, host='0.0.0.0', dev_tools_hot_reload=True)
