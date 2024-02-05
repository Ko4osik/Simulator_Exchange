import matplotlib.pyplot as plt


def plot_feature(
          feature, 
          include_exchanges : list= None, 
          exclude_exchanges : list = None, 
          separate_plots : list = False
):
    exchanges = feature.exchanges
    chosen_exchanges = dict()
    if (exclude_exchanges != None) and (include_exchanges != None):
        raise TypeError("Only one of exclude_exchanges and exclude_exchanges can be not None")
    elif exclude_exchanges != None:
        for exchange in exchanges:
                
                if exchange not in exclude_exchanges:
                    chosen_exchanges[exchange] = exchanges[exchange]
    elif include_exchanges != None:
        for exchange in exchanges:
                if exchange in include_exchanges:
                    chosen_exchanges[exchange] = exchanges[exchange]
    else:
        chosen_exchanges = exchanges.copy()


    if separate_plots == False:
        plt.figure(figsize = (14, 6))
        for exchange in list(chosen_exchanges.keys()):

            plt.plot(feature.compile_feature()[exchange], label = exchange)
        plt.title(f'{feature.get_feature_name()} plot')
        plt.xlabel('tick')
        plt.ylabel('value')
        plt.legend()
        plt.show()

    else:
        for i, exchange in enumerate(list(chosen_exchanges.keys())):
            plt.figure(figsize = (14, 2 * len(chosen_exchanges)))
            plt.subplot(len(chosen_exchanges), 1, i + 1)
            plt.plot(feature.compile_feature()[exchange])
            plt.title(f'{feature.get_feature_name()} plot on exchange {exchange}')
            plt.xlabel('tick')
            plt.ylabel('value')
            plt.show()

def plot_features_correlation(
    features : list          
):  
    if len(features) < 2:
        raise TypeError("The number of features to show can't be 0, add at least two features")
    exchanges = features[0].exchanges
    fig, axs = plt.subplots(len(features), len(features), figsize=(4 * len(features), 3 * len(features)))
    for exchange in exchanges:
        for i in range(len(features)):
            for j in range(len(features)):
                if i == j:
                    axs[i, i].hist(features[i].compile_feature()[exchange], bins = 10)
                else: 
                    axs[i, j].scatter(features[i].compile_feature()[exchange], (features[j].compile_feature()[exchange]))
