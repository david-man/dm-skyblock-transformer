DATA_TAGS = ['BOOSTER_COOKIE', 'ENCHANTED_DIAMOND_BLOCK', 'RECOMBOBULATOR_3000', 
            'KISMET_FEATHER', 'ENCHANTMENT_ULTIMATE_LEGION_1', 'ESSENCE_WITHER',
            'ESSENCE_CRIMSON', 'SUPER_COMPACTOR_3000', 
            'FLAWLESS_JADE_GEM', 'JACOBS_TICKET', 
            'SUMMONING_EYE', 'MAGMA_FISH_SILVER']
DATE_FORMAT = "%Y-%m-%d"
COLUMNS_OF_INTEREST = ['sell', 'sellVolume']#what features are you taking in
LABEL_COLUMN = 'sell'#what are you trying to predict for

PREDICTION_HORIZON = 4#how many days in advance of prediction

TRAINING_EPOCHS = 500
ENCODINGS = 4