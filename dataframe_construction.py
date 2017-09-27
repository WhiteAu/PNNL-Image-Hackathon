import pylidc as pl
import pandas as pd
import numpy as np

qu = pl.query(pl.Scan) #.filter(pl.Scan.scan_id <= 200)
data = pd.DataFrame(columns=['scan_id',
                             'patient',
                             'nodule_id',
                             'annotation',
                             'calcification',
                             'internalStructure',
                             'lobulation',
                             'malignancy',
                             'margin',
                             'sphericity',
                             'spiculation',
                             'subtlety',
                             'texture'])
scans = [j for (i, j) in enumerate(qu)]
i = -1

for scan in scans[:100]:

    annotations = [ann for ann in scan.annotations]

    for j, annotation in enumerate(annotations):
        i += 1
        data.loc[i, ['scan_id']] = annotation.scan_id
        data.loc[i, ['patient']] = scan.patient_id
        data.loc[i, ['nodule_id']]
        data.loc[i, ['annotation']] = j
        data.loc[i, ['calcification']] = annotation.calcification
        data.loc[i, ['internalStructure']] = annotation.internalStructure
        data.loc[i, ['lobulation']] = annotation.lobulation
        data.loc[i, ['malignancy']] = annotation.malignancy
        data.loc[i, ['margin']] = annotation.margin
        data.loc[i, ['sphericity']] = annotation.sphericity
        data.loc[i, ['spiculation']] = annotation.spiculation
        data.loc[i, ['subtlety']] = annotation.subtlety
        data.loc[i, ['texture']] = annotation.texture

