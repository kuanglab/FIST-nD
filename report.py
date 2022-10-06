import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import tempfile
import matplotlib.image as mpimg
import textwrap
from matplotlib.ticker import ScalarFormatter, FixedLocator
from matplotlib.colors import LogNorm


def fig_to_im(fig):
    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png", bbox_inches="tight", dpi=900)
        plt.close(fig)
        tmpfile.seek(0)
        return mpimg.imread(tmpfile.name)


def report(filename, metadata):
    titlesize = 30
    plt.rcParams["font.family"] = "serif"
    plt.style.use("fivethirtyeight")
    with PdfPages(filename) as pdf:
        # create a page for text summaries

        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.suptitle("Summary Statistics", fontsize=titlesize)
        text = ""
        text += f"Tensor shape: {metadata['shape']}\n"
        text += f"Number of validation spots: {metadata['val_spots']}\n"
        text += f"Genes kept: {metadata['n_genes']} of {metadata['og_genes']}"
        text += f" ({100*metadata['n_genes']/metadata['og_genes']:.1f}%)\n"
        text += f"Final MAE: {metadata['val_mae'][-1]:.5f}\n"
        text += f"Final MAPE: {metadata['val_smape'][-1]:.5f}\n"
        text += f"Final R²: {metadata['val_rmse'][-1]:.5f}\n"
        if "time_per_iter" in metadata:
            text += "Time per iteration: " + metadata["time_per_iter"] + "\n"
        text += str(metadata["arguments"])
        fig.text(
            0.1,
            0.9,
            "\n".join(textwrap.wrap(text, replace_whitespace=False)),
            transform=fig.transFigure,
            verticalalignment="top",
            wrap=True,
        )
        pdf.savefig(fig)

        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.suptitle("Highly Variable Genes", fontsize=titlesize)
        text = "\n".join(metadata["variable_genes"])
        fig.text(
            0.1,
            0.9,
            text,
            transform=fig.transFigure,
            verticalalignment="top",
            wrap=True,
        )
        pdf.savefig(fig)

        # plot validation metrics if available
        if "val_smape" in metadata.keys():
            val_mae = metadata["val_mae"]
            val_smape = metadata["val_smape"]
            val_rmse = metadata["val_rmse"]

            fig, axs = plt.subplots(3, figsize=(8.5, 11))
            fig.suptitle("Validation Metrics", fontsize=titlesize)
            axs[0].plot(val_mae)
            axs[1].plot(val_smape)
            axs[0].set_yscale("log")
            axs[1].set_yscale("log")
            axs[0].set_ylim(axs[0].get_ylim()[0], max(val_mae[len(val_mae) // 16 :]))
            axs[1].set_ylim(axs[1].get_ylim()[0], max(val_smape[len(val_smape) // 16 :]))
            axs[2].plot(val_rmse)
            axs[0].set_title("MAE (log scale)")
            axs[1].set_title("MAPE (log scale)")
            axs[2].set_title("R²")
            axs[0].yaxis.set_major_formatter(ScalarFormatter())
            axs[1].yaxis.set_major_formatter(ScalarFormatter())
            fig.tight_layout()
            pdf.savefig(fig)

        plt.set_cmap("plasma")
        # plot genes of interest
        for gene_to_plot in metadata["gene_plot"]:
            # create page
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(gene_to_plot, fontsize=titlesize)
            if not gene_to_plot + "_post" in metadata:
                fig.text(
                    0.1,
                    0.9,
                    f"Gene {gene_to_plot} was dropped in preprocessing and not predicted.",
                    transform=fig.transFigure,
                    verticalalignment="top",
                    wrap=True,
                )
                pdf.savefig(fig)
                plt.close(fig)
                continue

            coords = [
                [float(x) for x in coord.split("x")] for coord in metadata["coords"]
            ]
            coords = np.array(coords)

            exp_pre = np.array(metadata[gene_to_plot + "_pre"])
            exp_post = np.array(metadata[gene_to_plot + "_post"])
            vmin = min(min(exp_pre[np.nonzero(exp_pre)]), min(exp_post[np.nonzero(exp_post)]))
            vmax = max(max(exp_pre), max(exp_post))
            size = np.where(exp_pre > 0, 10, 0)

            # case for 2 spatial dimensions
            if metadata["shape"].count(",") == 2:
                # plot pre and post imputation at two different angles
                ax_pre_1_im = fig.add_subplot(2, 2, 1)
                ax_pre_2_im = fig.add_subplot(2, 2, 2)
                ax_post_1_im = fig.add_subplot(2, 2, 3)
                ax_post_2_im = fig.add_subplot(2, 2, 4)

                H = np.histogram2d(
                    *coords.T,
                    bins=np.max(coords, axis=0).astype(int),
                    weights=np.log(exp_pre),
                    normed=False,
                )[0].T
                ax_pre_1_im.imshow(H, vmin=0, vmax=vmax)
                ax_pre_1_im.grid(None)
                ax_pre_2_im.scatter(
                    *coords.T, c=np.log(exp_pre), s=size, vmin=0, vmax=vmax
                )
                H = np.histogram2d(
                    *coords.T,
                    bins=np.max(coords - 1e-6, axis=0).astype(int),
                    weights=np.log(exp_post),
                    normed=False,
                )[0].T
                ax_post_1_im.imshow(H, vmin=0, vmax=vmax)
                ax_post_1_im.grid(None)
                p = ax_post_2_im.scatter(
                    *coords.T, c=np.log(exp_post), s=10, vmin=0, vmax=vmax
                )

            # 3D plottting case
            elif metadata["shape"].count(",") == 3:

                ax_pre_1_im = fig.add_subplot(2, 2, 1)
                ax_pre_2_im = fig.add_subplot(2, 2, 2)
                ax_post_1_im = fig.add_subplot(2, 2, 3)
                ax_post_2_im = fig.add_subplot(2, 2, 4)

                ratio = np.ptp(coords, axis=0)

                # plot pre and post imputation at two different angles
                fig_small = plt.figure()
                ax_pre_1 = fig_small.add_subplot(projection="3d")
                ax_pre_1.scatter3D(
                    *coords.T, c=exp_pre, s=size, norm=LogNorm(vmin=vmin, vmax=vmax)
                )
                ax_pre_1.set_box_aspect(ratio)
                ax_pre_1_im.imshow(fig_to_im(fig_small))

                fig_small = plt.figure()
                ax_pre_2 = fig_small.add_subplot(projection="3d")
                ax_pre_2.scatter3D(
                    *coords.T, c=exp_pre, s=size, norm=LogNorm(vmin=vmin, vmax=vmax)
                )
                ax_pre_2.view_init(30, -120)
                ax_pre_2.set_box_aspect(ratio)
                ax_pre_2_im.imshow(fig_to_im(fig_small))

                fig_small = plt.figure()
                ax_post_1 = fig_small.add_subplot(projection="3d")
                ax_post_1.scatter3D(
                    *coords.T, c=exp_post, norm=LogNorm(vmin=vmin, vmax=vmax)
                )
                ax_post_1.set_box_aspect(ratio)
                ax_post_1_im.imshow(fig_to_im(fig_small))

                fig_small = plt.figure()
                ax_post_2 = fig_small.add_subplot(projection="3d")
                p = ax_post_2.scatter3D(
                    *coords.T, c=exp_post, norm=LogNorm(vmin=vmin, vmax=vmax)
                )
                ax_post_2.view_init(30, -120)
                ax_post_2.set_box_aspect(ratio)
                ax_post_2_im.imshow(fig_to_im(fig_small))

                axes = [ax_pre_1_im, ax_pre_2_im, ax_post_1_im, ax_post_2_im]
                for ax in axes:
                    ax.axis("off")
                    ax.grid(False)

            ax_pre_1_im.set_title("Pre-imputation")
            ax_pre_2_im.set_title("Pre-imputation")
            ax_post_1_im.set_title("Post-imputation")
            ax_post_2_im.set_title("Post-imputation")

            cbar_ax = fig.add_axes([0.05, 0.05, 0.9, 0.025])
            plt.colorbar(p, cax=cbar_ax, orientation="horizontal")
            pdf.savefig(fig)
            plt.close(fig)
        # Set the report metadata
        d = pdf.infodict()
        d["Title"] = "FIST Output"
        d["CreationDate"] = datetime.datetime.today()
        d["ModDate"] = datetime.datetime.today()
