import detectree as dtr
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import plot
from img_processing import close_mask_clusters
import maxflow as mf
import numpy as np


import pixel_features, pixel_response, settings, utils


MOORE_NEIGHBORHOOD_ARR = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])

class DetecTree(dtr.Classifier):
    def predict_img(self, img, *, img_cluster=None, output_filepath=None):
        clf = getattr(self, "clf", None)
        if clf is None:
            if img_cluster is not None:
                try:
                    clf = self.clf_dict[img_cluster]
                except KeyError:
                    raise ValueError(
                        f"Classifier for cluster {img_cluster} not found in"
                        " `self.clf_dict`."
                    )
            else:
                raise ValueError(
                    "A valid `img_cluster` must be provided for classifiers"
                    " instantiated with `clf_dict`."
                )
        return self._predict_img(img, clf, output_filepath=output_filepath)

    def _predict_img(self, src, clf, *, output_filepath=None):
        # ACHTUNG: Note that we do not use keyword-only arguments in this method because
        # `output_filepath` works as the only "optional" argument
        img_shape = src.shape[:2]


        X = pixel_features.PixelFeaturesBuilder(
            **self.pixel_features_builder_kwargs
        ).build_features_from_filepath(src)

        if not self.refine:
            y_pred = clf.predict(X).reshape(img_shape)
        else:
            p_nontree, p_tree = np.hsplit(clf.predict_proba(X), 2)
            g = mf.Graph[int]()
            node_ids = g.add_grid_nodes(img_shape)
            P_nontree = p_nontree.reshape(img_shape)
            P_tree = p_tree.reshape(img_shape)

            # The classifier probabilities are floats between 0 and 1, and the graph
            # cuts algorithm requires an integer representation. Therefore, we multiply
            # the probabilities by an arbitrary large number and then transform the
            # result to integers. For instance, we could use a `refine_int_rescale` of
            # `100` so that the probabilities are rescaled into integers between 0 and
            # 100 like percentages). The larger `refine_int_rescale`, the greater the
            # precision.
            # ACHTUNG: the data term when the pixel is a tree is `log(1 - P_tree)`,
            # i.e., `log(P_nontree)`, so the two lines below are correct
            D_tree = (self.refine_int_rescale * np.log(P_nontree)).astype(int)
            D_nontree = (self.refine_int_rescale * np.log(P_tree)).astype(int)
            # TODO: option to choose Moore/Von Neumann neighborhood?
            g.add_grid_edges(
                node_ids, self.refine_beta, structure=MOORE_NEIGHBORHOOD_ARR
            )
            g.add_grid_tedges(node_ids, D_tree, D_nontree)
            g.maxflow()
            # y_pred = g.get_grid_segments(node_ids)
            # transform boolean `g.get_grid_segments(node_ids)` to an array of
            # `self.tree_val` and `self.nontree_val`
            y_pred = np.full(img_shape, self.nontree_val)
            y_pred[g.get_grid_segments(node_ids)] = self.tree_val

        # TODO: make the profile of output rasters more customizable (e.g., via the
        # `settings` module)
        # output_filepath = path.join(output_dir,
        #                             f"tile_{tile_start}-{tile_end}.tif")
        if output_filepath is not None:
            with rio.open(
                output_filepath,
                "w",
                driver="GTiff",
                width=y_pred.shape[1],
                height=y_pred.shape[0],
                count=1,
                dtype=np.uint8,
                nodata=self.nontree_val,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                dst.write(y_pred.astype(np.uint8), 1)
        return y_pred

# def detectree_predict_plot(image, d, c, o):
#     y_pred = dtr.DetecTree().predict_img(image_path)
#     y_pred = dtr.Classifier().predict_imgs
#     mask = close_mask_clusters(y_pred, d, c, o)

#     # side-by-side plot of the tile and the predicted tree/non-tree pixels
#     figwidth, figheight = plt.rcParams["figure.figsize"]
#     fig, axes = plt.subplots(1, 2, figsize=(2 * figwidth, figheight))
#     with rio.open(image_path) as src:
#         plot.show(src, ax=axes[0])
#     axes[1].imshow(mask)