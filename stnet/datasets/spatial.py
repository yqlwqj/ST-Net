import os
import numpy as np
import pickle
import logging
import pathlib
import datetime
import time
import glob
import collections
import stnet
import tqdm
import pandas as pd

def spatial(args):
    import skimage.io

    window = 224  # only to check if patch is off of boundary
    stnet.utils.logging.setup_logging(args.logfile, args.loglevel)

    logger = logging.getLogger(__name__)

    pathlib.Path(args.dest).mkdir(parents=True, exist_ok=True)

    raw, subtype = load_raw(args.root)

    with open(args.dest + "/subtype.pkl", "wb") as f:
        pickle.dump(subtype, f)

    t = time.time()

    t0 = time.time()
    section_header = None
    gene_names = set()
    for patient in raw:
        for section in raw[patient]:
            section_header = raw[patient][section]["count"].columns.values[0]
            gene_names = gene_names.union(set(raw[patient][section]["count"].columns.values[1:]))
    gene_names = list(gene_names)
    gene_names.sort()
    with open(args.dest + "/gene.pkl", "wb") as f:
        pickle.dump(gene_names, f)
    gene_names = [section_header] + gene_names
    logger.info("Finding list of genes: " + str(time.time() - t0))

    for (i, patient) in enumerate(raw):
        logger.info("Processing " + str(i + 1) + " / " + str(len(raw)) + ": " + patient)

        for section in raw[patient]:
            pathlib.Path("{}/{}/{}".format(args.dest, subtype[patient], patient)).mkdir(parents=True, exist_ok=True)

            # This is just a blank file to indicate that the section has been completely processed.
            # Preprocessing occassionally crashes, and this lets the preparation restart from where it let off
            complete_filename = "{}/{}/{}/.{}".format(args.dest, subtype[patient], patient, section)
            if pathlib.Path(complete_filename).exists():
                logger.info("Patient {} section {} has already been processed.".format(patient, section))
            else:
                logger.info("Processing " + patient + " " + section + "...")

                # In the original data, genes with no expression in a section are dropped from the table.
                # This adds the columns back in so that comparisons across the sections can be done.
                t0 = time.time()
                missing = list(set(gene_names) - set(raw[patient][section]["count"].keys()))
                c = raw[patient][section]["count"].values[:, 1:].astype(float)
                pad = np.zeros((c.shape[0], len(missing)))
                c = np.concatenate((c, pad), axis=1)
                names = np.concatenate((raw[patient][section]["count"].keys().values[1:], np.array(missing)))
                c = c[:, np.argsort(names)]
                logger.info("Adding zeros and ordering columns: " + str(time.time() - t0))

                t0 = time.time()
                count = {}
                for (j, row) in raw[patient][section]["count"].iterrows():
                    count[row.values[0]] = c[j, :]
                logger.info("Extracting counts: " + str(time.time() - t0))

                t0 = time.time()
                tumor = {}
                not_int = False
                for (_, row) in raw[patient][section]["tumor"].iterrows():
                    if isinstance(row[1], float) or isinstance(row[2], float):
                        not_int = True
                    tumor[(int(round(row[1])), int(round(row[2])))] = (row[4] == "tumor")
                if not_int:
                    logger.warning("Patient " + patient + " " + section + " has non-integer patch coordinates.")
                logger.info("Extracting tumors: " + str(time.time() - t0))

                t0 = time.time()
                image = skimage.io.imread(raw[patient][section]["image"])
                logger.info("Loading image: " + str(time.time() - t0))

                data = []
                for (_, row) in raw[patient][section]["spot"].iterrows():
                    x = int(round(row['X']))
                    y = int(round(row['Y']))

#                   tumor_list = pd.DataFrame(list(tumor.items()))
                    tumor_list = pd.DataFrame(list(tumor.items()), columns=["1","2"])

                    X = image[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]

                    row_list = pd.DataFrame(list(row.items()))

                    if X.shape == (window, window, 3):
                        a = row["Unnamed: 0"].replace('x',', ')
                        a = '(' + a +')'
                        b = eval(a) 

                        if tuple(b) in tumor:
                            data.append((X,
#                                        count[str(int(row["x"])) + "x" + str(int(row["y"]))],
#              				 tumor[(int(row["x"]), int(row["y"]))],
#					 np.array([x, y]),
#                                        np.array([patient]),
#                                        np.array([section]),
#                                        np.array([int(row["x"]), int(row["y"])]),
#                                        ))
                                         count[row["Unnamed: 0"]],
              				 tumor[b],
					 np.array([x, y]),
                                         np.array([patient]),
                                         np.array([section]),
                                         np.array([int(row["X"]), int(row["Y"])]),
                                         ))


#                           filename = "{}/{}/{}/{}_{}_{}.npz".format(args.dest, subtype[patient], patient, section,
#                                                                     int(row["x"]), int(row["y"]))
                            filename = "{}/{}/{}/{}_{}_{}.npz".format(args.dest, subtype[patient], patient, section,
                                                                      int(row["X"]), int(row["Y"]))


#                           np.savez_compressed(filename, count=count[str(int(row["x"])) + "x" + str(int(row["y"]))],
#                                               tumor=tumor[(int(row["x"]), int(row["y"]))],
#                                               pixel=np.array([x, y]),
#                                               patient=np.array([patient]),
#                                               section=np.array([section]),
#                                               index=np.array([int(row["x"]), int(row["y"])]))
                            np.savez_compressed(filename, count=count[row["Unnamed: 0"]],
                                                tumor=tumor[b],
                                                pixel=np.array([x, y]),
                                                patient=np.array([patient]),
                                                section=np.array([section]),
                                                index=np.array([int(row["X"]), int(row["Y"])]))

                        else:
#                           logger.warning("Patch " + str(int(row["x"])) + "x" + str(
#                               int(row["y"])) + " not found in " + patient + " " + section)
                            logger.warning("Patch " + row["Unnamed: 0"]
                                 + " not found in " + patient + " " + section)
                    else:
                        logger.warning("Detected spot too close to edge.")
                logger.info("Saving patches: " + str(time.time() - t0))

                with open(complete_filename, "w"):
                    pass
    logger.info("Preprocessing took " + str(time.time() - t) + " seconds")

    if (not os.path.isfile(stnet.config.SPATIAL_PROCESSED_ROOT + "/mean_expression.npy") or
        not os.path.isfile(stnet.config.SPATIAL_PROCESSED_ROOT + "/median_expression.npy")):
        logging.info("Computing statistics of dataset")
        gene = []
        for filename in tqdm.tqdm(glob.glob("{}/*/*/*_*_*.npz".format(args.dest))):
            npz = np.load(filename)
            count = npz["count"]
            gene.append(np.expand_dims(count, 1))

        gene = np.concatenate(gene, 1)
        np.save(stnet.config.SPATIAL_PROCESSED_ROOT + "/mean_expression.npy", np.mean(gene, 1))
        np.save(stnet.config.SPATIAL_PROCESSED_ROOT + "/median_expression.npy", np.median(gene, 1))


def load_section(root: str, patient: str, section: str, subtype: str):
    """
    Loads data for one section of a patient.
    """
    import pandas
    import gzip

    #file_root = root + "/" + subtype + "/" + patient + "/" + patient + "_" + section
    file_root = root + subtype + "/" + patient + "/" + patient + "_" + section

    # image = skimage.io.imread(file_root + ".jpg")
    image = file_root + ".jpg"

    if stnet.utils.util.newer_than(file_root + ".tsv.gz", file_root + ".pkl"):
        with gzip.open(file_root + ".tsv.gz", "rb") as f:
            count = pandas.read_csv(f, sep="\t")
        with open(file_root + ".pkl", "wb") as f:
            pickle.dump(count, f)
    else:
        with open(file_root + ".pkl", "rb") as f:
            count = pickle.load(f)

    if stnet.utils.util.newer_than(file_root + ".spots.txt.gz", file_root + ".spots.pkl"):
        with gzip.open(file_root + ".spots.txt.gz", "rb") as f:
#           spot = pandas.read_csv(f, sep="\t")
            spot = pandas.read_csv(f, sep=",")
        with open(file_root + ".spots.pkl", "wb") as f:
            pickle.dump(spot, f)
    else:
        with open(file_root + ".spots.pkl", "rb") as f:
            spot = pickle.load(f)

    if stnet.utils.util.newer_than(file_root + "_Coords.tsv.gz", file_root + "_Coords.pkl"):
        with gzip.open(file_root + "_Coords.tsv.gz", "rb") as f:
            tumor = pandas.read_csv(f, sep="\t")
        with open(file_root + "_Coords.pkl", "wb") as f:
            pickle.dump(tumor, f)
    else:
        with open(file_root + "_Coords.pkl", "rb") as f:
            tumor = pickle.load(f)

    return {"image": image, "count": count, "spot": spot, "tumor": tumor}

def load_raw(root: str):
    """
    Loads data for all patients.
    """

    logger = logging.getLogger(__name__)

    # Wildcard search for patients/sections
    images = glob.glob(root + "/*/*/*_*.jpg")

    # Dict mapping patient ID (str) to a list of all sections available for the patient (List[str])
    patient = collections.defaultdict(list)
    for (p, s) in map(lambda x: x.split("/")[-1][:-4].split("_"), images):
        patient[p].append(s)

    # Dict mapping patient ID (str) to subtype (str)
    subtype = {}
    for (st, p) in map(lambda x: (x.split("/")[-3], x.split("/")[-1][:-4].split("_")[0]), images):
        if p in subtype:
            if subtype[p] != st:
                raise ValueError("Patient {} is marked as type {} and {}.".format(p, subtype[p], st))
        else:
            subtype[p] = st

    logger.info("Loading raw data...")
    t = time.time()
    data = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            data[p] = {}
            for s in patient[p]:
                data[p][s] = load_section(root, p, s, subtype[p])
                pbar.update()
    logger.info("Loading raw data took " + str(time.time() - t) + " seconds.")

    return data, subtype
