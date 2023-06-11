import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html,dcc,dash_table,no_update
import pandas as pd
import numpy as np
from apricot import FacilityLocationSelection
from apricot import MaxCoverageSelection
from apricot import GraphCutSelection
from apricot import SaturatedCoverageSelection
from apricot import FeatureBasedSelection
from submodlib.functions.facilityLocation import FacilityLocationFunction
import dash
import io
import dash_bootstrap_components as dbc
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from sklearn import datasets
from sklearn.cluster import KMeans
import sys
import base64
import os
from umap import UMAP
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
import io
import time
import dash_uploader as du
import uuid
from pathlib import Path
from sklearn.manifold import TSNE
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from openicl import DatasetReader
from openicl import PromptTemplate
from openicl import RandomRetriever
from openicl import TopkRetriever
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from openicl import DatasetReader, PromptTemplate, TopkRetriever, PPLInferencer,ZeroRetriever,SubmodRetriever, RandomRetriever
import dash_auth



iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])


models = ['distilgpt2','facebook/opt-125m','bigscience/bloom-560m']
clusters = ['yes','no']
retriever = ['Random','Prompt','Facility','DPP','Graph-Cut','concave']
Inferencer = ['Channel','Perplexity','COT']
optimizer = ['NAIVE','LAZY','TWO-STAGE','APPROXIMATE','STOCHASTIC','MODULAR','BIDIRECTIONAL']
projcharts= ['BERT','CNN','REGRESSION']
similarity = ['EUCLIDEAN','COSINE']
projmodels= ['PCA','UMAP','TSNE']
demonstration_number = [1,2,3,4,5,6,7,8,9,10]


UPLOAD_DIRECTORY = "app_uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:

try:
   os.makedirs("file_uploaded")
   
except FileExistsError:
   # directory already exists
   pass
UPLOAD_FOLDER_ROOT = 'file_uploaded'
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

auth = dash_auth.BasicAuth(
    app,
    {'root': '1234',
     'paul': 'paul'}
)


du.configure_upload(app,UPLOAD_FOLDER_ROOT)

def get_upload_component(id):
    return du.Upload(

        id=id,
        text="Drop",
        text_completed="Completed: ",
        cancel_button=True,
        pause_button=True,
        max_file_size=1000000,  # 130 Mb
        filetypes=["csv", "zip"],
        upload_id=uuid.uuid1(),  # Unique session id
        max_files=1,
    )


visualise_selection = dbc.Card(
    [

        html.Div(
            [
                dbc.Label("VISUALISE SUBSET"),
                dcc.Dropdown(
                    id="proj-selection",
                    options=[
                        {"label": col, "value": col} for col in projmodels
                    ],
                    value="PCA",
                ),
 
            ]
        ),
    ]
)

retriever_model_select = dbc.Card(
    [
          html.Div(
            [
                dbc.Label("Model"),
                dcc.Dropdown(
                    id="retriever-model",
                    options=[
                        {"label": col, "value": col} for col in retriever
                    ],
                    value="Random",
                ),
 
            ]
        ),
    ]
)


demonstration_example_selection = dbc.Card(
    [
            html.Div(
            [
                dbc.Label("Demonstration Examples"),
                dcc.Dropdown(
                    id="number-selected",
                    options=[
                        {"label": col, "value": col} for col in demonstration_number 
                    ],
                    value=1,
                ),
            ]
        ),

    
    ]
)


inference_model_selection = dbc.Card(
    [

        html.Div(
            [
                dbc.Label("Inference Model"),
                dcc.Dropdown(
                    id="inferencer-model",
                    options=[
                        {"label": col, "value": col} for col in models 
                    ],
                    value="distilgpt2",
                ),
            ]
        ),
    ]
)
inference_method_selection = dbc.Card(
    [

        html.Div(
            [
                dbc.Label("Generation Method"),
                dcc.Dropdown(
                    id="inferencer",
                    options=[
                        {"label": col, "value": col} for col in Inferencer
                    ],
                    value="Perplexity",
                ),
            ]
        ),
    ]
)
classification_metric = dbc.Card(
    [
    html.Div(
    [
    dbc.Label("Classification Metrics"),
    html.P(id = 'accuracy'),
    html.P(id='f1-score'),
    html.P(id='recall'),
    html.P(id='precision'),
  
    ]
    )
    ]
)

tab2 = dbc.Card(
    [
        html.Div(
            [
               dbc.Button("RUN", id="button-run"),
            ]
        ),
        html.Div(
            [
                dbc.Label("SELECTION PROGRESS"),
                html.Div(id="time-taken"),

            ]
        ),
    ],
    body=True,
)


tab1 = dbc.Card(
    [

    html.Div(
    [
    dbc.Label("Write prompt"),
    dbc.Textarea(
    id='prompt-text',
    value='Write a prompt for the texts, For example for text classification task, you could say \n This article is about: \n The sentiment of the text is:\nMake sure you delete these instruction and writte you prompt before submiting',
    style={"width": "100%", "height": "30vh"}
    ),
    dbc.Button('Submit', id='write-prompt'),
    html.Div(id='submit-feedback'),
    
    ]
    ),

    html.Div(
    [
     dash_table.DataTable(
        id='datatable-upload-container',
        data = [],
        columns = [],
    style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    },
        
        )
    
  
        

    ]
    ),

    ]
)

file_uploiad = dbc.Card(
    [
    html.Div(
        [
    dbc.Label("Upload File"),
        get_upload_component(id='dash-uploader'),
         html.Div(id='callback-output'),
         
        ],

    ),

    ]
)

tab_subset_viz = dbc.Tab(label="TRAIN DATA", tab_id="tab-models",children = [

dbc.Card(
    [

        html.Div(
            [
                dbc.Label("VISUALISE TRAINING DATASET "),
                dcc.Dropdown(
                    id="proj-selection",
                    options=[
                        {"label": col, "value": col} for col in projmodels
                    ],
                    value="PCA",
                ),
 
            ]
        ),
    ]
)
]


)

tab_full_viz = dbc.Tab(label="TEST DATA", tab_id="full-data-viz",children = [

dbc.Card(
    [

        html.Div(
            [
                dbc.Label("VISUALISE PREDICTED TEST LABELS"),
                dcc.Dropdown(
                    id="proj-selection-full",
                    options=[
                        {"label": col, "value": col} for col in projmodels
                    ],
                    value="PCA",
                ),
 
            ]
        ),
    ]
)
]


)

app.layout = dbc.Container(
    fluid=True,
    children=[
    html.Div(
    [
        dbc.Button("INTERACTIVE IN-CONTEXT LEARNING", color="primary"),
    ],
    className="d-grid gap-2",
),
        html.Hr(),
        dbc.Row(
    [
    dbc.Col(
    children = [
    file_uploiad,
  
    ],
    width=6,
    lg=3
    
    ),
     dbc.Col(
    children = [
    retriever_model_select
    
  
    ],
    ),
     dbc.Col(
    children = [
    demonstration_example_selection
    
  
    ]
    ),
dbc.Col(
    children = [
    inference_model_selection
    
  
    ]
    ),
     dbc.Col(
    children = [
    inference_method_selection
    
  
    ]
    ),

    ],
    className="g-0"
        ),
        dbc.Row(
            [
                dbc.Col(
                    width=7,
                    children=[
                        dcc.Store(id='intermediate-value',data=[], storage_type='memory'),
                        dcc.Store(id='hover-data-test',data=[], storage_type='memory'),
                        dcc.Store(id='hover-data-train',data=[], storage_type='memory'),
                        dcc.Store(id='train-data',data=[], storage_type='memory'),
                
                tab1,
                    ],
                ),
                dbc.Col(
                    width=5,
                    children=[
                         dbc.Tabs([tab_subset_viz, tab_full_viz], id='tabs', active_tab='tab_full_viz'),
                         
                        dcc.Graph(id='subset-viz'),
                        dcc.Tooltip(id="graph-tooltip"),


                         dbc.Card(
                            body=True,
                            children=[
                        classification_metric,
    
                            ],
                        ),

                    ],
                ),
            ]
        ),
    ],
)


@app.callback(
   [Output('intermediate-value', 'data'),
    Output('datatable-upload-container', 'data'),
    Output('datatable-upload-container', 'columns'),
    Output('submit-feedback', 'children'),
    Output('train-data', 'data'),
    ],
    #Output('subset-viz', 'figure'),
    [
    #Input('tabs', 'active_tab'),
   Input('write-prompt', 'n_clicks'),
    #State('random-seed', 'value')
    #Input('update-button', 'n_clicks')
    #Input("btn_csv", "n_clicks")
    ],
    [
    State('dash-uploader', 'isCompleted'),
    State('dash-uploader', 'fileNames'),
     State('dash-uploader', 'upload_id'),
    State('prompt-text', 'value'),
    State('retriever-model', 'value'),
     State('number-selected', 'value'),
     State('inferencer-model', 'value'),
     #State('percentage-selected', 'value'),
     #State('cluser-data', 'value'),
     #State('cluster-number', 'value'),
     
     ],
    prevent_initial_call=True,
)
def callback_on_completion(prompt_run,  iscompleted,filenames, upload_id, prompt_text,retriver,demonstrations, inferencer):

    out = []
    texts = []
    labels = []
    if filenames is not None:
        if upload_id:
            root_folder = Path(UPLOAD_FOLDER_ROOT) / upload_id
        else:
            root_folder = Path(UPLOAD_FOLDER_ROOT)

        for filename in filenames:
            file = root_folder / filename
            out.append(file)
            df = pd.read_csv(file)
            text = df['text'].tolist()
            label = df['label'].tolist()
            for i in text:
                texts.append(i)
            for j in label:
                labels.append(j)

    data_analysis = pd.DataFrame()
    data_analysis ['text'] =  texts
    data_analysis ['label'] = labels

    f = open('out.txt', 'w')
    f.close()

    if iscompleted and prompt_run >0:
        
        orig_stdout = sys.stdout
        f = open('out.txt', 'a')
        sys.stdout = f
        
        decription = prompt_text

        num_demonstation = demonstrations
        train,test = train_test_split(data_analysis , test_size=0.3)
        validation, test = train_test_split(test, test_size=0.5)
        ds_dict = {'train' : Dataset.from_pandas(train),
                   'validation': Dataset.from_pandas(validation),
           'test' : Dataset.from_pandas(test )}
        dataset = DatasetDict(ds_dict)
        data = DatasetReader(dataset, input_columns=['text'], output_column='label')

        unique_labels =  data_analysis["label"].unique()

        tp_dict = {label: f"</E>Text: <X>\n{decription}: {label}" for label in unique_labels }
        template = PromptTemplate(tp_dict, column_token_map={'text' : '<X>'}, ice_token='</E>')
        if retriver == "Prompt":
             
             
             retriever = ZeroRetriever(data)
             inferencer = PPLInferencer(model_name=inferencer)
             predictions = inferencer.inference(retriever, ice_template=template)


        elif retriver == "Random":

            retriever = RandomRetriever(data, ice_num=num_demonstation)
            inferencer = PPLInferencer(model_name=inferencer)
            predictions = inferencer.inference(retriever, ice_template=template)

        elif retriver == "Facility":
            retriever =  SubmodRetriever(data, ice_num=num_demonstation, batch_size=8, candidate_num=30,submodular_function='facility_variant')
            inferencer = PPLInferencer(model_name=inferencer)
            predictions = inferencer.inference(retriever, ice_template=template)


        elif retriver == "Graph-Cut":
            retriever =  SubmodRetriever(data, ice_num=num_demonstation, batch_size=8, candidate_num=30,submodular_function='graph_cut')
            inferencer = PPLInferencer(model_name=inferencer)
            predictions = inferencer.inference(retriever, ice_template=template)


        elif retriver == "DPP":
            retriever =  SubmodRetriever(data, ice_num=num_demonstation, batch_size=8, candidate_num=30,submodular_function='dpp')
            inferencer = PPLInferencer(model_name=inferencer)
            predictions = inferencer.inference(retriever, ice_template=template)


        elif retriver == "concave":
            retriever =  SubmodRetriever(data, ice_num=num_demonstation, batch_size=8, candidate_num=30,submodular_function='concave')
            inferencer = PPLInferencer(model_name=inferencer)
            predictions = inferencer.inference(retriever, ice_template=template)
        


        df_prompt = pd.DataFrame()
        df_prompt['text'] = test['text'].tolist()
        df_prompt['true_labels'] = test['label'].tolist()
        df_prompt['labels'] = predictions
        sample = df_prompt.sample(n=4)
        columns = [{"name": i, "id": i} for i in sample.columns]
        train_df = pd.DataFrame()
        train_df['text'] = train['text'].tolist()
        train_df['labels'] = train['label'].tolist()

        sys.stdout = orig_stdout
        f.close()

        file = open('out.txt', 'r')
        verbose_model_output =''
        lines = file.readlines()
        if lines.__len__()<=3:
            last_lines=lines
        else:
            last_lines = lines[-3:]
        for line in last_lines:
            verbose_model_output = verbose_model_output+line + '<BR>'
        file.close()




      





        return  df_prompt.to_json(orient='split'),sample.to_dict('records'),columns,decription,  train_df.to_json(orient='split')
    






@app.callback(Output('subset-viz', 'figure'),
 Output('accuracy', 'children'),
 Output('f1-score', 'children'),
 Output('recall', 'children'),
 Output('precision', 'children'),
 Output('hover-data-train', 'data'),
 Input('proj-selection', 'value'),
 State('intermediate-value', 'data'),
 State('train-data', 'data'),
 Input('tabs','active_tab'),
prevent_initial_call=True,)
def update_graph(proj_viz, subset_to_plot, train_data, tab):
    dff_viz = pd.read_json(subset_to_plot, orient='split')
    y_viz = dff_viz['labels'].tolist()
    texts = dff_viz['text'].tolist()
    true_labels = dff_viz['true_labels'].tolist()
    accuracy = accuracy_score(true_labels, y_viz)
    f1 = f1_score(true_labels, y_viz, average='macro')
    precision = precision_score(true_labels, y_viz, average='macro')
    recall = recall_score(true_labels, y_viz, average='macro')
  
    labels = y_viz
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedings = model.encode( texts )
    x_viz= embedings 
    kmeans_model = KMeans(n_clusters=3, random_state=1).fit(x_viz)

    train_viz = pd.read_json(train_data, orient='split')
    train_y = train_viz['labels'].tolist()
    texts_train = train_viz['text'].tolist()
    embeddings_train =  model.encode(texts_train)



    acc =f"Accuracy: {accuracy*100:.2f}%"
    f1_scores = f"f1-score: {f1*100:.2f}%"
    prec = f"Precision: { precision*100:.2f}%"
    rec =  f"Recall: { recall*100:.2f}%"
    
    if tab and subset_to_plot is not None:
        if tab == 'tab-models':
            if proj_viz == 'TSNE':
                tsne = TSNE(n_components=2, verbose=1, random_state=123)
                z = tsne.fit_transform( embeddings_train) 
                df_tsne_train = pd.DataFrame(z)
                df_tsne_train  =   df_tsne_train.rename(columns={0:'x',1:'y'})
                df_tsne_train  =   df_tsne_train.assign(label= train_viz.labels.values)
                df_tsne_train  =   df_tsne_train.assign(text = train_viz.text.values)
                silhourte = metrics.silhouette_score(z, train_y, metric='euclidean')
                fig = px.scatter(
                df_tsne_train ,
                x = "x",
                y = "y",
                color = 'label',
                title =  f"Silhourte Coeficient for the train is: {silhourte:.4f}"
            )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_yaxes(visible=False, showticklabels=False)
                fig.update_xaxes(visible=False, showticklabels=False)
                fig.update_layout(showlegend=False)
                fig.update_coloraxes(showscale=False)
                fig.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'})
            elif proj_viz == 'PCA':
                pca = PCA(n_components=2)
                z =  pca.fit_transform( embeddings_train) 
                df_tsne_train  = pd.DataFrame(z)
                df_tsne_train  =  df_tsne_train.rename(columns={0:'x',1:'y'})
                df_tsne_train  =  df_tsne_train.assign(label= train_viz.labels.values)
                df_tsne_train  =  df_tsne_train.assign(text = train_viz.text.values)
                silhourte = metrics.silhouette_score(z, train_y, metric='euclidean')
                fig = px.scatter(
                df_tsne_train ,
                x = 'x',
                y = 'y',
                color = 'label',
                title =  f"Silhourte Coeficient for the train is:: {silhourte:.4f}"
            )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_yaxes(visible=False, showticklabels=False)
                fig.update_xaxes(visible=False, showticklabels=False)
                fig.update_layout(showlegend=False)
                fig.update_coloraxes(showscale=False)
                fig.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'})
                
            elif proj_viz == 'UMAP':
                umap = UMAP(n_components=2)
                z =  umap.fit_transform(embeddings_train) 
                df_tsne_train  = pd.DataFrame(z)
                df_tsne_train  =  df_tsne_train.rename(columns={0:'x',1:'y'})
                df_tsne_train  =  df_tsne_train.assign(label= train_viz.labels.values)
                df_tsne_train  =  df_tsne_train .assign(text = train_viz.text.values)
                silhourte = metrics.silhouette_score(z, train_y, metric='euclidean')
                fig = px.scatter(
                df_tsne_train ,
                x = 'x',
                y = 'y',
                color = 'label',
                title =  f"Silhourte Coeficient for the train is:: {silhourte:.4f}"
            )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_yaxes(visible=False, showticklabels=False)
                fig.update_xaxes(visible=False, showticklabels=False)
                fig.update_layout(showlegend=False)
                fig.update_coloraxes(showscale=False)
                fig.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'})
            
            return fig,  acc, f1_scores,  rec, prec, df_tsne_train.to_json(orient='split')
                

        elif tab == 'full-data-viz':
            if proj_viz == 'TSNE':
                tsne = TSNE(n_components=2, verbose=1, random_state=123)
                z = tsne.fit_transform(x_viz) 
                df_tsne = pd.DataFrame(z)
                df_tsne = df_tsne.rename(columns={0:'x',1:'y'})
                df_tsne = df_tsne.assign(label= dff_viz.labels.values)
                df_tsne = df_tsne.assign(text = dff_viz.text.values)
                silhourte = metrics.silhouette_score(z, y_viz, metric='euclidean')
                fig = px.scatter(
                df_tsne,
                x = 'x',
                y = 'y',
                color = 'label',
                title =  f"Silhourte Coeficient for the test is:: {silhourte:.4f}"
            )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_yaxes(visible=False, showticklabels=False)
                fig.update_xaxes(visible=False, showticklabels=False)
                fig.update_layout(showlegend=False)
                fig.update_coloraxes(showscale=False)
                fig.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'})
            elif proj_viz == 'PCA':
                pca = PCA(n_components=2)
                z =  pca.fit_transform(x_viz) 
                df_tsne = pd.DataFrame(z)
                df_tsne = df_tsne.rename(columns={0:'x',1:'y'})
                df_tsne = df_tsne.assign(label= dff_viz.labels.values)
                df_tsne = df_tsne.assign(text = dff_viz.text.values)
                silhourte = metrics.silhouette_score(z, y_viz, metric='euclidean')
                fig = px.scatter(
                df_tsne,
                x = 'x',
                y = 'y',
                color = 'label',
                title =  f"Silhourte Coeficient for the test is:: {silhourte:.4f}"
            )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_yaxes(visible=False, showticklabels=False)
                fig.update_xaxes(visible=False, showticklabels=False)
                fig.update_layout(showlegend=False)
                fig.update_coloraxes(showscale=False)
                fig.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'})
                
            elif proj_viz == 'UMAP':
                umap = UMAP(n_components=2)
                z =  umap.fit_transform(x_viz) 
                df_tsne = pd.DataFrame(z)
                df_tsne = df_tsne.rename(columns={0:'x',1:'y'})
                df_tsne = df_tsne.assign(label= dff_viz.labels.values)
                df_tsne = df_tsne.assign(text = dff_viz.text.values)
                silhourte = metrics.silhouette_score(z, y_viz, metric='euclidean')
                fig = px.scatter(
                df_tsne,
                x = 'x',
                y = 'y',
                color = 'label',
                title =  f"Silhourte Coeficient for the test is:: {silhourte:.4f}"
            )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                fig.update_yaxes(visible=False, showticklabels=False)
                fig.update_xaxes(visible=False, showticklabels=False)
                fig.update_layout(showlegend=False)
                fig.update_coloraxes(showscale=False)
                fig.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'})
            return fig,  acc, f1_scores,  rec, prec, df_tsne.to_json(orient='split')
    return "Data still processing or not uploaded"


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    State('train-data', 'data'),
    Input("subset-viz", "hoverData"),
)
def display_hover(hover_df, hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    dff_hover = pd.read_json(hover_df, orient='split')
    df_row = dff_hover.iloc[num]
    texts = df_row['text']
    label = df_row['labels']

    children = [
        html.Div([
            html.H2(f"{label}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
            html.P(f"{texts}")

        ],style={'width': '200px', 'white-space': 'normal'}
        )
        
    ]

    return True, bbox, children






if __name__ == '__main__':
    app.run_server(debug=True,port=8063)

