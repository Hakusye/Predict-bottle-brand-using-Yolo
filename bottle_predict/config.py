configurations = {
    "config":{
    "SEED" : 1234,
    "DATA_LOOT" : "/",
    "ROOT_IMAGES_PATH":"../self_images/",
    "EXT":"png",
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 224,
    "DEVICE" : "cuda",
    "CLASSES" : 16,
    "MEAN" : (0.485,0.456,0.406),
    "STD" : (0.299,0.224,0.225),
    "MAX_EPOCH" : 20,
    "WEIGHT_DECAY" : 1e-3
    },

    "class_name" :{
       "aquarius" : 0,
       "ayataka_brown" : 1,
       "calpis": 2,
        "namacha" : 3,
       "natural_green" : 4,
        "irohas" : 5,
        "gogo_tea":6,
       "tropicana" : 7,
       "ooi_ocha" : 8,
       "coca_cola" : 9,
       "ayataka" : 10,
       "dekavita" : 11,
       "pokari" : 12,
       "iemon" : 13,
       "genmai" : 14,
       "koicha" : 15
    },

    "rev_class_name" :[
        "aquarius",
        "ayataka_browm",
        "calpis",
        "namacha",
        "natural_green",
        "irohas",
        "gogo_tea",
        "tropicana",
       "ooi_ocha",
       "coca_cola",
       "ayataka",
       "dekavita",
       "pokari",
       "iemon",
       "genmai",
       "koicha"
    ]
}