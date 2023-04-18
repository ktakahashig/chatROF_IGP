import pandas as pd
import openai
import ast
import streamlit as st



openai.api_key  = st.secrets["OPENAI_API_KEY"]
#with open('/Users/ken/.openai.key') as f:
#    openai.api_key  = f.read().rstrip() 

def get_embed_interpretaciones(df):
    from openai.embeddings_utils import get_embedding
    df['embeddings'] = [get_embedding(x,engine="text-embedding-ada-002") for x in df.Combinado]
    return df

def semantic_search(df, query):
    from openai.embeddings_utils import get_embedding,cosine_similarity
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
    df=df.sort_values('similarity', ascending=False).reset_index()
    df = df.drop(columns=['embeddings','index','Unnamed: 0'])
    return df

def responder_consulta(results,query):
    import openai
    # Preparar lista de Referencias :: Resultados y conclusiones para el prompt
    #dum=results.sort_values(by='similarity',ascending=False).drop_duplicates()
    dum=results[['Artículo','Numeral','Combinado']].drop_duplicates()
    dum['Artículo'] = dum['Artículo'].astype(str)+dum['Numeral'].astype(str).str.replace("nan","")
    dum=dum.drop(columns='Numeral')
    lista_resultados = ''
    for index, row in dum.iterrows():
        lista_resultados += ' :: '.join(row.astype(str)) + '\n'    
        
    message_sys = """
        Eres un experto legal que asesora sobre documentos de gestión institucionales.
        """
    message_user = """
        Responde la consulta sobre "<Tema>" basado en los siguientes artículos del Reglamento de 
        Organización y Funciones (ROF) del Instituto Geofísico del Perú (IGP), proporcionados
        en el formato Artículo :: Detalle. Debes hacer referencia a los artículos 
        específicos sobre los cuales basas la respuesta a la consulta en la forma "(Art. 3j, 24)", 
        por ejemplo.
        
    Artículo :: Detalle
    <Referencias>
    
    """
    message_assist = """
    Respuesta a la consulta <Tema>:
    """
    message_user = message_user.replace("<Tema>", query)
    message_user = message_user.replace("<Referencias>", lista_resultados)
    message_assist = message_assist.replace("<Tema>", query)
    messages = [{"role": "system", "content": message_sys},{"role": "user", "content": message_user},
                {"role": "assistant", "content": message_assist}]

    # Correr el modelo y obtener el reporte
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, max_tokens=2000, 
                                            n=1, stop=None, temperature=0.0)
    report = response['choices'][0]['message']['content'].strip()
    
    return report


rof = pd.read_csv('5_rof_embeddings.csv')
rof['embeddings']=rof.embeddings.apply(lambda s: list(ast.literal_eval(s)))


st.title('ChatROF-IGP')
st.caption('Consulta el ROF del IGP usando inteligencia artificial')

query = st.text_input("Ingrese su consulta: " )

if query != '':
   with st.spinner('Espere mientras la IA genera el reporte ...'):
      result = semantic_search(rof, query)
      respuesta= responder_consulta(result[:10],query)
      st.success("Respuesta a consulta sobre: "+query)
      st.write(respuesta)
      dum=result[['Artículo','Numeral','Tema','Combinado','similarity']][:10].reset_index()
      dum = dum.drop(columns=['index'])
      dum['Ref']=dum['Artículo'].astype(str)+dum['Numeral'].astype(str).str.replace("nan","")
      dum = dum.sort_values('Numeral').sort_values('Artículo')
      dum = dum[['Ref','Tema','Combinado','similarity']]
      st.write('\nReferencias:')
      for i in range(len(dum)):
         st.write("Art. "+dum.Ref[i]+". "+dum.Tema[i]+"<br>"+dum.Combinado[i]+"<br>"
             +"(sim. "+"{:.3f}".format(dum.similarity[i])+")<br>", unsafe_allow_html=True)
         #st.write(dum.Combinado[i])
         #st.write("(sim. "+"{:.3f}".format(dum.similarity[i])+")\n")


st.caption('© Ken Takahashi Guevara, 2023')

