from flask import Flask, render_template, request, jsonify
import pymysql
import pyaudio
import numpy as np
import tensorflow as tf
import threading
import queue
import keyboard
import time
import torch
import esc_config as config
from model.htsat import HTSAT_Swin_Transformer
import keyboard
app = Flask(__name__)

def get_db_connection():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='bd_sonidos_peligrosos',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection

class Audio_Classification:
    def __init__(self, model_path, config):
        super().__init__()
        self.device = torch.device('cuda')
        self.sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )
        ckpt = torch.load(model_path, map_location="cpu")
        temp_ckpt = {}
        for key in ckpt["state_dict"]:
            temp_ckpt[key[10:]] = ckpt['state_dict'][key]
        self.sed_model.load_state_dict(temp_ckpt)
        self.sed_model.to(self.device)
        self.sed_model.eval()


    def predict(self, waveform):
        if waveform.size > 0:
            with torch.no_grad():
                x = torch.from_numpy(waveform).float().to(self.device)
                output_dict = self.sed_model(x[None, :], None, True)
                pred = output_dict['clipwise_output']
                pred_post = pred[0].detach().cpu().numpy()
                pred_label = np.argmax(pred_post)
                pred_prob = np.max(pred_post)
            return pred_label, pred_prob

# Configuraciones de audio
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 32000
CHUNK = 1024  # Cambiado para igualar a tu configuración anterior
SILENCE_THRESHOLD = 0.03  # Umbral para considerar un sonido como tal
SEGMENT_DURATION = 3  # Duración de cada segmento de audio en segundos

# Clase para clasificar los audios (ya definida en tu código)
audiocls = Audio_Classification('workspace/results/exp_htsat_dataset_sonidos_peligrosos/checkpoint/lightning_logs/version_4/checkpoints/l-epoch=49-acc=0.905.ckpt', config)


# Cola para comunicación entre hilos
audio_queue = queue.Queue()

# Estado de predicción activado/desactivado
prediccion_activada = True
audio_interface = pyaudio.PyAudio()
stream = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Función para procesar y clasificar el audio en un hilo separado en tiempo real
def audio_processing_thread(model):
    global prediccion_activada
    try:
        audio_buffer = []
        while True:
            
            data = stream.read(CHUNK)
            numpy_data = np.frombuffer(data, dtype=np.float32)
            audio_buffer.append(numpy_data)

            # Verificar si hemos acumulado suficientes datos para 5 segundos de audio
            if len(audio_buffer) * CHUNK >= RATE * SEGMENT_DURATION:
                concatenated_data = np.concatenate(audio_buffer)
                audio_buffer = []  # Limpiar el buffer para el siguiente segmento de audio

                # Verificar silencio antes de predecir
                if prediccion_activada and np.sqrt(np.mean(np.square(concatenated_data))) > SILENCE_THRESHOLD:
                    pred_label, pred_prob = audiocls.predict(concatenated_data)
                    audio_queue.put((pred_label, pred_prob))
                else:
                    print("Silencio detectado, omitiendo predicción.")
    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

# Iniciar el hilo de procesamiento de audio
threading.Thread(target=audio_processing_thread, args=(audiocls,), daemon=True).start()



@app.route('/')
def index():
    return render_template('Frontend/monitoreo_tiempo_real.html')

@app.route('/estadisticas')
def statistics():
    return render_template('Frontend/estadisticas.html')

clasesPredecir = {0: 'llorar', 1: 'cristal_roto', 2: 'disparos', 3: 'fuego', 4: 'fuegos_artificiales', 5: 'fuga_gas', 6: 'gritos', 7: 'llorar', 8: 'lluvia', 9: 'tormenta_electrica'}

@app.route('/get_audio_data', methods=['GET'])
def get_audio_data():
    if not audio_queue.empty():

        pred_label, pred_prob =  audio_queue.get()
        pred_prob = round(float(pred_prob), 2)
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # La fecha se establece automáticamente en la base de datos con NOW()
            sql = "INSERT INTO `sonidos_detectados` (`nro_microfono`, `clase_sonido`, `confianza`, `fecha`) VALUES (%s, %s, %s, NOW())"
            cursor.execute(sql, (1, clasesPredecir[pred_label], pred_prob))
            connection.commit()
        connection.close()
        
        print(clasesPredecir[pred_label], pred_prob)

        return jsonify({'class': clasesPredecir[pred_label], 'score': pred_prob})
    return jsonify({'class': 'none', 'score': 0.0})


@app.route('/total-sonidos-detectados')
def total_sonidos_detectados():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM sonidos_detectados')
    result = cur.fetchone()
    total = result['COUNT(*)'] if result else 0
    cur.close()
    conn.close()
    return jsonify(total=total)

@app.route('/total-clases-detectadas')
def total_clases_detectadas():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(DISTINCT clase_sonido) FROM sonidos_detectados')
    result = cur.fetchone()
    total_clases = result['COUNT(DISTINCT clase_sonido)'] if result else 0
    cur.close()
    conn.close()
    return jsonify(total_clases=total_clases)

@app.route('/total-promedio-confianza')
def total_promedio_confianza():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT AVG(confianza) FROM sonidos_detectados')
    result = cur.fetchone()
    promedio_confianza = round(result['AVG(confianza)'], 2) if result['AVG(confianza)'] is not None else 0
    cur.close()
    conn.close()
    return jsonify(promedio_confianza=promedio_confianza)



@app.route('/datos')
def datos():
    clase_sonido = request.args.get('claseSonido', default="")
    top_n = request.args.get('topN', default=None, type=int)

    conn = get_db_connection()
    cur = conn.cursor()

    base_query = 'SELECT clase_sonido, COUNT(*) as numero_de_detecciones FROM sonidos_detectados'
    where_conditions = []
    params = []

    if clase_sonido:
        where_conditions.append('clase_sonido = %s')
        params.append(clase_sonido)

    query = base_query
    if where_conditions:
        query += ' WHERE ' + ' AND '.join(where_conditions)
    query += ' GROUP BY clase_sonido ORDER BY numero_de_detecciones DESC'

    if top_n:
        query += ' LIMIT %s'
        params.append(top_n)

    cur.execute(query, params)
    sonidos_detectados = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(sonidos_detectados)



@app.route('/promedio-confianza')
def promedio_confianza():
    clase_sonido = request.args.get('claseSonido', default=None)
    confianza_min = request.args.get('confianzaMin', type=float, default=0)
    confianza_max = request.args.get('confianzaMax', type=float, default=1)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Construye la consulta SQL con los filtros
    query = 'SELECT clase_sonido, AVG(confianza) AS promedio_confianza FROM sonidos_detectados'
    where_clauses = []
    params = []
    
    if clase_sonido:
        where_clauses.append('clase_sonido = %s')
        params.append(clase_sonido)
        
    where_clauses.append('confianza >= %s')
    params.append(confianza_min)
    
    where_clauses.append('confianza <= %s')
    params.append(confianza_max)
    
    if where_clauses:
        query += ' WHERE ' + ' AND '.join(where_clauses)
    
    query += ' GROUP BY clase_sonido ORDER BY promedio_confianza DESC'

    cur.execute(query, params)
    resultado = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(resultado)


@app.route('/incidentes-tiempo')
def incidentes_tiempo():
    fecha_inicio = request.args.get('fechaInicio', default=None)
    fecha_fin = request.args.get('fechaFin', default=None)
    query = '''
        SELECT DATE(fecha) as fecha, COUNT(*) as cantidad
        FROM sonidos_detectados
    '''
    where_clauses = []
    params = []
    
    if fecha_inicio:
        where_clauses.append('DATE(fecha) >= %s')
        params.append(fecha_inicio)
    
    if fecha_fin:
        where_clauses.append('DATE(fecha) <= %s')
        params.append(fecha_fin)
    
    if where_clauses:
        query += ' WHERE ' + ' AND '.join(where_clauses)
    
    query += ' GROUP BY DATE(fecha) ORDER BY DATE(fecha)'
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    resultados = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(resultados)

@app.route('/sonidos-distribucion')
def sonidos_distribucion():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT clase_sonido, COUNT(*) as cantidad
        FROM sonidos_detectados
        GROUP BY clase_sonido
    ''')
    resultados = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(resultados)

@app.route('/datos-por-dia')
def datos_por_dia():
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT DAYNAME(fecha) as dia_semana, COUNT(*) as cantidad
            FROM sonidos_detectados
            GROUP BY DAYNAME(fecha)
            ORDER BY FIELD(DAYNAME(fecha), 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday');
        """)
        resultado = cursor.fetchall()
    connection.close()
    return jsonify(resultado)

@app.route('/datos-por-hora')
def datos_por_hora():
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT HOUR(fecha) as hora, COUNT(*) as cantidad
            FROM sonidos_detectados
            GROUP BY HOUR(fecha)
            ORDER BY HOUR(fecha);
        """)
        resultado = cursor.fetchall()
    connection.close()
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
