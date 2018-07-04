# The interactive graph is based on: https://medium.com/@gorjanz/data-analysis-in-python-interactive-scatterplot-with-matplotlib-6bb8ad2f1f18

import random as rnd
import time
import tensorflow as tf
import sys
sys.path.append("../../data")
sys.path.append("../readers")
sys.path.append("../lm_cond")
sys.path.append("../tm")
from reader_ptb import Reader_PTB
from reader_news20 import Reader_News20
from reader_apnews import Reader_APNEWS
from RNN import tmGRUCell, tmLSTMCell
from LDA import LDA
from LM_cond_output import LOG, LM

import numpy as np
import pickle
from sklearn.manifold import TSNE
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.text import Annotation
import seaborn as sns
import matplotlib.cm as cm

def test_plot(data, max_data_points=1000):
    sns.set()

    # # define two colors, just to enrich the example
    # labels_color_map = {0: '#20b2aa', 1: '#ff7373'}
    #
    # # set the examples count
    # no_examples = 50
    #
    # # generate the data needed for the scatterplot
    # generated_data = [(x, rnd.randint(0, no_examples)) for x in range(0, no_examples)]
    # generated_labels = ["Label for instance #{0}".format(i) for i in range(0, no_examples)]
    #
    data_pc = defaultdict(lambda: [])
    for i in range(len(data[:max_data_points])):
        data_pc[ data[i]["target"] ].append( {"sentence": data[i]["sentence"], "tsne": data[i]["tsne"]} )



    colors = iter(cm.rainbow(np.linspace(0, 1, len(data_pc.items()))))

    labels_color_map = {}
    topic_to_label = {}
    for i, (k,v) in enumerate( data_pc.items()):
        labels_color_map[ i ] = next(colors)
        # print(i,k,v)
        topic_to_label[k] = i

    instances_colors = []
    axis_values_x = []
    axis_values_y = []
    labels = []

    for i in range(len(data[:max_data_points])):
        instances_colors.append( labels_color_map[ topic_to_label[ data[i]["target"]]] )
        axis_values_x.append( data[i]['tsne'][0] )
        axis_values_y.append( data[i]['tsne'][1] )
        labels.append( " ".join( data[i]["sentence"]))



    print("now visualizing scatterlplot...")


    # draw a scatter-plot of the generated values
    fig = plt.figure(figsize=(20, 16))
    ax = plt.subplot()


    for k,v in data_pc.items():
        instances_colors_t = []
        axis_values_x_t = []
        axis_values_y_t = []
        labels_t = []
        for val in v:
            instances_colors_t.append( labels_color_map[ topic_to_label[ k] ])
            axis_values_x_t.append( val['tsne'][0] )
            axis_values_y_t.append( val['tsne'][1] )
        ax.scatter(
            axis_values_x_t,
            axis_values_y_t,
            c=instances_colors_t,
            label=k,
            alpha=0.5,
            picker=True
        )


    # extract the scatterplot drawing in a separate function so we ca re-use the code
    def draw_scatterplot():
        ax.scatter(
            axis_values_x,
            axis_values_y,
            c=instances_colors,
            alpha=0.5,
            picker=True
        )


    # draw the initial scatterplot
    # draw_scatterplot()


    # create and add an annotation object (a text label)
    def annotate(axis, text, x, y):
        text_annotation = Annotation(text, xy=(x, y), xycoords='data')
        axis.add_artist(text_annotation)


    # define the behaviour -> what happens when you pick a dot on the scatterplot by clicking close to it
    def onpick(event):
        # step 1: take the index of the dot which was picked
        ind = event.ind

        # step 2: save the actual coordinates of the click, so we can position the text label properly
        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata

        # just in case two dots are very close, this offset will help the labels not appear one on top of each other
        offset = 0

        # if the dots are to close one to another, a list of dots clicked is returned by the matplotlib library
        for i in ind:
            # step 3: take the label for the corresponding instance of the data
            label = labels[i]

            # step 4: log it for debugging purposes
            print("index", i, label)

            # step 5: create and add the text annotation to the scatterplot
            annotate(
                ax,
                label,
                label_pos_x + offset,
                label_pos_y + offset
            )

            # step 6: force re-draw
            ax.figure.canvas.draw_idle()

            # alter the offset just in case there are more than one dots affected by the click
            offset += 0.01

    # connect the click handler function to the scatterplot
    fig.canvas.mpl_connect('pick_event', onpick)

    # create the "clear all" button, and place it somewhere on the screen
    ax_clear_all = plt.axes([0.0, 0.0, 0.1, 0.05])
    button_clear_all = Button(ax_clear_all, 'Clear all')


    # define the "clear all" behaviour
    def onclick(event):
        # step 1: we clear all artist object of the scatter plot
        ax.cla()

        # step 2: we re-populate the scatterplot only with the dots not the labels
        draw_scatterplot()

        # step 3: we force re-draw
        ax.figure.canvas.draw_idle()


    # link the event handler function to the click event on the button
    button_clear_all.on_clicked(onclick)

    # initial drawing of the scatterplot

    plt.plot()
    print("scatterplot done")
    ax.legend()
    # present the scatterplot
    plt.show()




def tsne_plot(data, max_data_points=1000, title="Placeholder Title"):
    data_pc = defaultdict(lambda: [])
    for i in range(len(data[:max_data_points])):
        data_pc[ data[i]["target"] ].append( {"sentence": data[i]["sentence"], "tsne": data[i]["tsne"]} )
    plt.figure(figsize=(20, 20))
    for k,v in data_pc.items():
        print("k", k, "len v", len(v))
        x = [x["tsne"][0] for x in v]
        y = [x["tsne"][1] for x in v]
        plt.scatter(x, y,  alpha=0.3, label=k)
    plt.legend()
    plt.title(title)

    plt.show()


def add_latent_vector(dataModel, model, savedParameters, batch_size=64, first_x=1000):
    with tf.Session() as sess:
        sess.run(model.init_op)
        model.all_saver.restore(sess, savedParameters)

        new_data = []
        for i, doc in enumerate(dataModel.train_data[:first_x]):
            temp_new = {"target" : doc["target"]}
            prepped_batch, SL = dataModel.create_experiment_batch( doc["sentences"], batch_size=batch_size )
            z_temp = sess.run([ model.z], feed_dict={model.X_in: prepped_batch,
                                                     model.inference: True,
                                                     model.seq_lengths: SL,
                                                     model.keep_prob: 1.0 })
            temp_new["sentences"] = doc["sentences"]
            temp_new["z"] = z_temp[:len(doc["sentences"])][0]
            new_data.append(temp_new)
            if i%100 == 0:
                sys.stdout.flush()
                print("[", i, ":", len(dataModel.train_data), "]" , flush=True)


    return new_data

# def latent_sentence_level(DocLevelData, batch_size=64):
#     data = []
#     for i, d in enumerate(DocLevelData):
#         for j,s in enumerate(d["sentences"]):
#             # print(i, j, np.shape(d["z"][j]))
#
#             if j >= batch_size:
#                 break
#             data.append( {"target": d["target"],
#                           "sentence": s,
#                           "z": DocLevelData[i]["z"][j]
#                           })
#     return data

# def apply_tsne(data):
#     X = [x["z"] for x in data]
#     print("dimensions X for t-sne:", np.shape(X))
#     print("start t-sne...")
#     start = time.time()
#     X_embedded = TSNE(n_components=2).fit_transform(X)
#     end = time.time()
#     elapsed =   round( end - start, 1)
#     print("performed t-sne in", elapsed)
#     for i, x in enumerate(data):
#         data[i]["tsne"] = X_embedded[i]
#     return data


def create_t_sne_data( lm, sample_size = 400):
    data = lm.reader.train[:sample_size]
    new_data = []
    for d in data:
        z = lm.tm.get_topic_distribution(  [np.array(d["doc_tm"])]  )[0]

        sample = " ".join( [ lm.reader.lm_id2word[x]   for x in lm.do_sample( seed="", num_samples=10, topic=z)])

        new_data.append( {"z": z,
                          "sentence": sample,
                          'target': "Document representation"})

    X = [x["z"] for x in new_data]
    print("dimensions X for t-sne:", np.shape(X))
    print("start t-sne...")
    start = time.time()
    X_embedded = TSNE(n_components=2).fit_transform(X)
    end = time.time()
    elapsed =   round( end - start, 1)
    print("performed t-sne in", elapsed)
    for i, x in enumerate(new_data):
        new_data[i]["tsne"] = X_embedded[i]

    return new_data

def interpolate(n=5):
    vecs = []
    for i, x in enumerate(range(n)):
        temp = np.zeros((20))
        temp[8] = 1 - (i/n)
        temp[16] =  i * 1.0 / n
        vecs.append(temp)

    return vecs







if __name__ == "__main__":

    reader_apnews = pickle.load(open("../saved/apnews.p", "rb"))
    # reader_news20 = pickle.load(open("../saved/news_20.p", "rb"))

    with tf.Session() as sess:
        print("latentSpaceAnalysis is run as __main__:")
        print("start training tm apnews")
        # tm = LDA(reader_apnews , topics=20, max_iter=50 )
        # pickle.dump(tm, open("tm.p", "wb"))
        tm = pickle.load(open("tm.p", "rb"))
        tm.experiments( save_location="results/AN_2_tm_topics_med_1_keep.txt")

        print("start training lm")
        m = LM(session=sess,
               reader=reader_apnews,
               topic_model=tm,
               num_layers=2,
               logger_tm_prog= LOG(file_path="results/AN_2_lm_topic_progression_med_1_keep.txt"),
               logger_results = LOG(file_path="results/AN_2_results_med_1_keep"),
               savepath_best = "results/AN_2_best_topic_med_300_1_keep.txt",
               embedding_size=300,
               hidden_size = 300,
               max_gradients=5,
               keep_prob=1.0,
               lr_decay=0.8)

        # Save the LM

        print("vocab size:",  reader_apnews.lm_vocab_size)

        m.initalize_from_trained_model(saved_model_path="temp/language_model.ckpt" )
        # m.print_topic_sample()
        # print("initialized the topics and stuff")
        # # m.train_model(epochs=40,save_path_lm_model="temp/language_model.ckpt", )
        #
        # t_sne_data = create_t_sne_data( lm=m)
        # pickle.dump( t_sne_data, open("tsne_data.p", "wb"))
        # t_sne_data = pickle.load( open( "tsne_data.p", "rb"))
        # print(t_sne_data[0])
        # test_plot(t_sne_data)
        # tsne_plot(t_sne_data)


        #Gimme a gradient
        interpolation  = interpolate(7)
        for z in interpolation:
            sample = " ".join( [ m.reader.lm_id2word[x]   for x in m.do_sample( seed="", num_samples=10, topic=z)])
            print(sample)


    # vectorized = add_latent_vector(reader_apnews, LM,  "savedModels/VAE.ckpt")
    # pickle.dump(vectorized, open("z_per_doc.p", "wb"))
    # vectorized = pickle.load(open("z_per_doc.p", "rb"))
    # data = latent_sentence_level(vectorized)
    # t_sne_data = apply_tsne(data)
    # print("save t-sne data...")
    # pickle.dump( t_sne_data, open("tsne_data", "wb"))
    #
    # t_sne_data = pickle.load( open( "tsne_data", "rb"))
    # print(t_sne_data[0])
    # test_plot(t_sne_data)
    # tsne_plot(t_sne_data)
