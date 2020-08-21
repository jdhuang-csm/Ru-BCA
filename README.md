# Fast-ramping Ru/(BaO)-(CaO)-(Al<sub>2</sub>O<sub>3</sub>) catalysts for ammonia synthesis
This repository contains data, code, and analysis for a study of Ru/(BaO)-(CaO)-(Al<sub>2</sub>O<sub>3</sub>) catalysts for ammonia synthesis. A journal article describing the full study in detail is in preparation.

## Methods
* (BaO)<sub>x</sub>-(CaO)<sub>y</sub>-(Al<sub>2</sub>O<sub>3</sub>)<sub>1-x-y</sub> (BCA) catalyst supports were synthesized via sol-gel and solid-state methods. 
* Ru/BCA catalysts were prepared via incipient wetness impregnation to achieve 1 wt. % Ru loading.
* Ammonia synthesis rates were measured for Ru/BCA catalysts in a packed-bed reactor at 480-600<sup>o</sup>C, 10 bar, and space velocities up to 1.3 x 10<sup>6</sup> ml g<sub>cat</sub><sup>-1</sup> h<sup>-1</sup>.
* A machine learning (ML) model was employed to improve understanding of the behavior of Ru/BCA catalysts.
* A microkinetic model was developed to describe the best-performing catalyst.

## Key Results
* A novel catalyst, Ru/(BaO)<sub>2</sub>(CaO)(Al<sub>2</sub>O<sub>3</sub>) (Ru/B2CA), was identified, which delivers ammonia synthesis rates as high as 189 mmol g<sub>cat</sub><sup>-1</sup> h<sup>-1</sup>
* ML analysis revealed a distinct volcano relationship between ammonia formation rate and a specialized descriptor, the mean (metal-oxygen)-(metal-nitrogen) bond energy delta, which serves as a proxy for nitrogen adsorption energy on the BCA support. This relationship is highly reminiscent of the volcano relationship between turnover frequency and nitrogen adsoprtion energy for unsupported metal catalysts.
* The volcano relationship suggests that BCA supports with optimal (metal-oxygen)-(metal-nitrogen) bond energy deltas are able to boost synthesis rates by adsorbing nitrogen directly, removing the nitrogen adsorption bottleneck observed for typical ammonia synthesis catalysts.
* The microkinetic model for Ru/B2CA revealed that (1) nitrogen adsorption is not the rate-limiting step in ammonia synthesis over Ru/B2CA and (2) the Ru surface is rich with adsorbed nitrogen, lending support to the mechanistic hypothesis derived from the ML model.

![AdsMech](/images/SynthesisCartoon_annote.PNG)*Proposed mechanism for ammonia formation on Ru/B2CA.*

## Repository Resources
The following resources are available in this repository:
* All data used for the ML analysis described above ([data](https://github.com/jdhuang-csm/Ru-BCA/tree/master/data))
* Python modules for generating phyiscochemical descriptors for Ru/BCA catalysts, ML model evaluation and selection, and visualization ([modules](https://github.com/jdhuang-csm/Ru-BCA/tree/master/modules))
* Jupyter notebooks to reproduce all data processing, analysis, and figure creation ([notebooks](https://github.com/jdhuang-csm/Ru-BCA/tree/master/notebooks))
* Figures and images generated by notebooks ([images](https://github.com/jdhuang-csm/Ru-BCA/tree/master/images))
* Citations for Python packages used (note that these were auto-generated and contain some errors) ([citations](https://github.com/jdhuang-csm/Ru-BCA/tree/master/citations))
