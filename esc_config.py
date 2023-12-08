exp_name = "exp_htsat_dataset_sonidos_peligrosos" # nombre del prefijo del ckpt guardado del modelo
workspace = "./workspace" # la carpeta de tu código
dataset_path = "./workspace/dataset_sonidos_peligrosos" # la ruta del conjunto de datos
desed_folder = "" # el archivo desed

dataset_type = "sonidos peligrosos" 

loss_type = "clip_ce" # es el tipo de pérdida, puede ser clip_ce, clip_bce, asl_loss


# ruta entrenado desde un punto de control, o evaluar un solo modelo
resume_checkpoint = "./workspace/ckpt/htsat_audioset_pretrain.ckpt"

 
esc_fold = 0 # solo para el conjunto de datos esc, selecciona el pliegue que necesitas para evaluación y (+1) validación


debug = False

random_seed = 970131 # 19970318 970131 12412 127777 1009 34047
batch_size = 32 # tamaño del lote por GPU x número de GPU, por defecto es 32 x 4 = 128
learning_rate = 1e-3 # 1e-4 también funciona
max_epoch = 50
num_workers = 0

lr_scheduler_epoch = [10,20,30]
lr_rate = [0.02, 0.05, 0.1]

# estas optimizaciones de preparación de datos no traen muchas mejoras, por lo que están obsoletas
enable_token_label = False # token label
class_map_path = "class_hier_map.npy"
class_filter = None 
retrieval_index = [15382, 9202, 130, 17618, 17157, 17516, 16356, 6165, 13992, 9238, 5550, 5733, 1914, 1600, 3450, 13735, 11108, 3762, 
    9840, 11318, 8131, 4429, 16748, 4992, 16783, 12691, 4945, 8779, 2805, 9418, 2797, 14357, 5603, 212, 3852, 12666, 1338, 10269, 2388, 8260, 4293, 14454, 7677, 11253, 5060, 14938, 8840, 4542, 2627, 16336, 8992, 15496, 11140, 446, 6126, 10691, 8624, 10127, 9068, 16710, 10155, 14358, 7567, 5695, 2354, 8057, 17635, 133, 16183, 14535, 7248, 4560, 14429, 2463, 10773, 113, 2462, 9223, 4929, 14274, 4716, 17307, 4617, 2132, 11083, 1039, 1403, 9621, 13936, 2229, 2875, 17840, 9359, 13311, 9790, 13288, 4750, 17052, 8260, 14900]
token_label_range = [0.2,0.6]
enable_time_shift = False # desplazamiento de tiempo
enable_label_enhance = False # mejorar la etiqueta jerárquica
enable_repeat_mode = False # repetir el espectrograma / remodelar el espectrograma



# para el diseño del modelo
enable_tscam = True # habilitar la capa token-semántica

# para el procesamiento de señales
sample_rate = 32000 
clip_samples = sample_rate * 10 # clip de 10 segundos 
window_size = 1024
hop_size = 320 
mel_bins = 64 
mel_bins = 64
fmin = 50
fmax = 14000
shift_max = int(clip_samples * 0.5)


# para la recolección de datos
classes_num = 10 
patch_size = (25, 4) # obsoleto
crop_size = None # int(clip_samples * 0.5) obsoleto

# para hiperparámetros htsat
htsat_window_size = 8
htsat_spec_size =  256
htsat_patch_size = 4 
htsat_stride = (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 96 
htsat_depth = [2,2,6,2]

swin_pretrain_path = None

# Algunas Optimizaciones Obsoletas en el diseño del modelo, revisar el código del modelo para detalles
htsat_attn_heatmap = False
htsat_hier_output = False 
htsat_use_max = False


# para la prueba del conjunto 

ensemble_checkpoints = []
ensemble_strides = []