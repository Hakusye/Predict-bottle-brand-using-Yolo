configurations = {
    "config":dict(
    SEED = 1234,
    DATA_LOOT = "/",
    BATCH_SIZE = 32,
    IMAGE_SIZE = 224,
    DEVICE = "cuda",
    CLASSES = 13,
    mean = (0.485,0.456,0.406),
    std = (0.299,0.224,0.225),
    MAX_EPOCH = 30,
    WEIGHT_DECAY = 1e-3
    ),

    "class_name" :{
        "aquarius" : 0,
        "ayataka_brown_rice" : 1,
        "calpis": 2,
        "craft_boss_black" : 3,
        "craft_boss_latte" : 4,
        "crystal_geyser" : 5,
        "fresh_tea" : 6,
        "green_dataka" : 7,
        "irohas" : 8,
        "sprite" : 9,
        "toropicana" : 10,
        "wilkinson" : 11,
    }
}