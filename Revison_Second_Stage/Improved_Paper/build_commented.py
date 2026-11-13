import difflib
import os
import re

BASE_DIR = "/home/smartlab/Documents/Research/BARD/PINN/PINN_CODE_PAPER/Revison_Second_Stage/Improved_Paper"
ORIG_PATH = os.path.join(BASE_DIR, "03_24_2026_paper_original.tex")
CLEAN_PATH = os.path.join(BASE_DIR, "PINN_revised_paper_clean.tex")
COMM_PATH = os.path.join(BASE_DIR, "PINN_revised_commented.tex")

with open(CLEAN_PATH, "r", encoding="utf-8") as f:
    text = f.read()

preamble_target = r"\usepackage{placeins}%"
preamble_repl = (
    r"\usepackage{placeins}%" + "\n"
    r"\usepackage{marginnote}%" + "\n"
    r"\setlength{\marginparwidth}{1.8cm}%" + "\n"
    r"\renewcommand*{\marginfont}{\tiny\sffamily\bfseries\color{blue}}%" + "\n"
    r"\newcommand{\revtag}[1]{\marginnote{#1}}%"
)
assert text.count(preamble_target) == 1
text = text.replace(preamble_target, preamble_repl, 1)


def tag_once(src, target, replacement, label):
    cnt = src.count(target)
    assert cnt == 1, f"Mismatch for {label}: count={cnt}"
    return src.replace(target, replacement, 1)


# 1. Abstract (all changes relative to 03_24_2026_paper_original.tex colored blue)
text = tag_once(
    text,
    r"analytical models require site-specific constitutive calibration, while unconstrained data-driven methods can violate physical sign and monotonicity laws needed for reliable predictive control decisions.",
    r"analytical models {\color{blue}require site-specific constitutive calibration, while unconstrained data-driven methods can violate physical sign and monotonicity laws needed for reliable predictive} control decisions.",
    "Abstract challenge",
)
text = tag_once(
    text,
    r"modeling flow through physically regularized neural branches that yield ordinary differential equations at sensor depths. Trained in two stages—derivative-matching initialization followed by closed-loop fourth-order Runge--Kutta (RK4) trajectory refinement—the model predicts soil moisture across six depths (5--55~\si{cm}) in a commercial Florida citrus block with an overall 3-day held-out forecast RMSE of \SI{1.23}{\%VWC} ($R^2 = 0.807$, Pearson $r = 0.921$, $p = \num{7.94e-178}$; \SI{3.22}{mm} integrated over the 60-\si{cm} profile) while simultaneously inferring depth-resolved evapotranspirative depletion. Across 30 rolling 3-day unseen windows, a 7-day horizon sensitivity check, and physical Jacobian admissibility audits, the proposed PINN reduces 3-day forecast error by 35.6\% over persistence, outperforms single-stage and unregularized ablations, and eliminates the unphysical control-sign inversions (33\%--97\%) exhibited by unconstrained linear and black-box Neural ODE baselines, providing an interpretable model suitable for future model-predictive irrigation control.",
    r"modeling flow through {\color{blue}physically regularized} neural branches that yield ordinary differential equations at sensor depths. Trained {\color{blue}in two stages—derivative-matching initialization followed by closed-loop fourth-order Runge--Kutta (RK4) trajectory refinement—}the model predicts soil moisture {\color{blue}across six depths (5--55~\si{cm}) in a commercial Florida citrus block with an overall 3-day held-out forecast RMSE of \SI{1.23}{\%VWC} ($R^2 = 0.807$, Pearson $r = 0.921$, $p = \num{7.94e-178}$; \SI{3.22}{mm} integrated over the 60-\si{cm} profile) while simultaneously inferring depth-resolved evapotranspirative depletion. Across 30 rolling 3-day unseen windows, a 7-day horizon sensitivity check, and physical Jacobian admissibility audits, the proposed PINN reduces 3-day forecast error by 35.6\% over persistence, outperforms single-stage and unregularized ablations, and eliminates the unphysical control-sign inversions (33\%--97\%) exhibited by unconstrained linear and black-box Neural ODE baselines, providing an interpretable model suitable for future model-predictive irrigation control.}",
    "Abstract body",
)

# 2. Introduction
text = tag_once(
    text,
    r"Agriculture consumes 70\% of global freshwater\cite{scanlon2023},",
    r"\marginnote{E-1.1, R-1.3a, R-1.3d}Agriculture consumes 70\% of global freshwater\cite{scanlon2023},",
    "Intro abstract marginnote",
)
text = tag_once(
    text,
    r"Throughout this work, soil moisture (SM) and soil water content (SWC) are quantified on a volumetric basis as volumetric water content (\si{\%VWC}), defined as the volume of liquid water per bulk volume of soil ($\mathrm{cm^3\,cm^{-3}} \times 100$); for a \SI{10}{cm} vertical soil compartment, \SI{1}{\%VWC} corresponds directly to \SI{1}{mm} of equivalent water depth. Potential atmospheric evaporative demand is quantified by the reference evapotranspiration $ET_0$ (\si{mm/h}), computed from meteorological measurements via the FAO-56 Penman--Monteith equation~\cite{allen1998crop}. Smart irrigation systems, which utilize sensors and algorithms to optimize water delivery, offer a solution; however, while recent machine-learning, time-series, and water-balance frameworks have begun integrating historical SWC, weather variables, and uncertainty estimates for multi-depth forecasting and irrigation scheduling~\cite{Zou2010, Bagheri2025, condeforecast2024}, developing multi-depth models that simultaneously preserve physical consistency (e.g., non-negative drainage and monotonic forcing response) and retain direct structural suitability for model predictive control (MPC) remains challenging.",
    r"\marginnote{R-1.1, R-1.2, R-1.3a}\textcolor{blue}{Throughout this work, soil moisture (SM) and soil water content (SWC) are quantified on a volumetric basis as volumetric water content (\si{\%VWC}), defined as the volume of liquid water per bulk volume of soil ($\mathrm{cm^3\,cm^{-3}} \times 100$); for a \SI{10}{cm} vertical soil compartment, \SI{1}{\%VWC} corresponds directly to \SI{1}{mm} of equivalent water depth. Potential atmospheric evaporative demand is quantified by the reference evapotranspiration $ET_0$ (\si{mm/h}), computed from meteorological measurements via the FAO-56 Penman--Monteith equation~\cite{allen1998crop}.} Smart irrigation systems, which utilize sensors and algorithms to optimize water delivery, offer a solution; however, \textcolor{blue}{while recent machine-learning, time-series, and water-balance frameworks have begun integrating historical SWC, weather variables, and uncertainty estimates for multi-depth forecasting and irrigation scheduling~\cite{Zou2010, Bagheri2025, condeforecast2024}, developing multi-depth models that simultaneously preserve physical consistency (e.g., non-negative drainage and monotonic forcing response) and retain direct structural suitability for model predictive control (MPC) remains challenging.}",
    "Intro Para 1",
)
text = tag_once(
    text,
    r"\cite{zhang2021}, whereas control-oriented MHE/MPC formulations~\cite{condeforecast2024} typically aggregate the root zone into a lumped single-zone water-balance compartment, and existing multi-depth data-assimilation or sequence-learning hybrids~\cite{Zou2010, Jiang2025, Jiang2026} omit explicit irrigation actuation ODE branches.",
    r"\cite{zhang2021}\marginnote{R-1.1}\textcolor{blue}{, whereas control-oriented MHE/MPC formulations~\cite{condeforecast2024} typically aggregate the root zone into a lumped single-zone water-balance compartment, and existing multi-depth data-assimilation or sequence-learning hybrids~\cite{Zou2010, Jiang2025, Jiang2026} omit explicit irrigation actuation ODE branches.}",
    "Intro Move 2 MPC",
)
text = tag_once(
    text,
    r'Purely data-driven machine learning models can predict soil moisture accurately in-distribution, but they produce "black box" outputs that can violate sign and monotonicity conditions with respect to control inputs ($q_{\mathrm{irr}}, ET_0$) and lack the physical consistency and stability guarantees necessary for model predictive control.',
    r'Purely data-driven machine learning models can predict soil moisture \marginnote{R-1.3d}\textcolor{blue}{accurately in-distribution}, but they produce "black box" outputs that \textcolor{blue}{can violate sign and monotonicity conditions with respect to control inputs ($q_{\mathrm{irr}}, ET_0$) and} lack the physical consistency and stability guarantees necessary for model predictive control.',
    "Intro Move 2 ML",
)
text = tag_once(
    text,
    r"More recently, constitutive neural networks have been embedded in differentiable finite-volume Backward Euler solvers with Newton root-finding on fine grids ($100$ cells on a \SI{10}{cm} lab column; $150$ cells on a \SI{1.5}{m} synthetic profile) \cite{Bandai2024}, and physics-informed operator networks (PI-DeepONet) have mapped 100-day meteorological series to $(z,t)$ profiles \cite{Ye2025}, alongside multi-depth time-series models \cite{Zou2010}, physics-guided LSTM/CNN-LSTM predictors \cite{WangY2023, ZhangT2025, Jiang2026}, EnKF--\texttt{fsolve} assimilation \cite{Jiang2025}, and lumped water-balance MHE/MPC \cite{condeforecast2024} (Table~\ref{tab:lit_comparison}). However, fine-grid implicit PDE surrogates and operator networks omit crop-sink and irrigation actuation channels, sequence networks retain black-box recurrent states, and lumped MPC models lack vertical root-zone resolution. Addressing this gap, the present study contributes: (1) a \emph{sensor-depth finite-volume discretization} of Richards' equation aligned directly with commercial multi-depth probes (5--55~\si{cm}) without fixed van Genuchten/Brooks--Corey curves; (2) \emph{decomposed, physically regularized neural flux branches} ($c_1$--$c_3$) for vertical drainage, signed inter-layer exchange, lateral ditch outflow, rainfall, irrigation, and evapotranspirative depletion; and (3) \emph{two-stage derivative-to-RK4 identification} verified through multi-window benchmarks, ablations, and Jacobian control-admissibility audits.",
    r"\marginnote{R-1.1, R-2.1}\textcolor{blue}{More recently, constitutive neural networks have been embedded in differentiable finite-volume Backward Euler solvers with Newton root-finding on fine grids ($100$ cells on a \SI{10}{cm} lab column; $150$ cells on a \SI{1.5}{m} synthetic profile) \cite{Bandai2024}, and physics-informed operator networks (PI-DeepONet) have mapped 100-day meteorological series to $(z,t)$ profiles \cite{Ye2025}, alongside multi-depth time-series models \cite{Zou2010}, physics-guided LSTM/CNN-LSTM predictors \cite{WangY2023, ZhangT2025, Jiang2026}, EnKF--\texttt{fsolve} assimilation \cite{Jiang2025}, and lumped water-balance MHE/MPC \cite{condeforecast2024} (Table~\ref{tab:lit_comparison}). However, fine-grid implicit PDE surrogates and operator networks omit crop-sink and irrigation actuation channels, sequence networks retain black-box recurrent states, and lumped MPC models lack vertical root-zone resolution. Addressing this gap, the present study contributes: (1) a \emph{sensor-depth finite-volume discretization} of Richards' equation aligned directly with commercial multi-depth probes (5--55~\si{cm}) without fixed van Genuchten/Brooks--Corey curves; (2) \emph{decomposed, physically regularized neural flux branches} ($c_1$--$c_3$) for vertical drainage, signed inter-layer exchange, lateral ditch outflow, rainfall, irrigation, and evapotranspirative depletion; and (3) \emph{two-stage derivative-to-RK4 identification} verified through multi-window benchmarks, ablations, and Jacobian control-admissibility audits.}",
    "Intro Move 4",
)
text = tag_once(
    text,
    r"""\begin{table}[!htbp]
\centering
\caption{Positioning of the proposed sensor-depth PINN relative to representative physics-informed soil hydrology and control-oriented smart irrigation studies.}
\label{tab:lit_comparison}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.165\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.175\textwidth}>{\raggedright\arraybackslash}p{0.185\textwidth}@{}}""",
    r"""\begin{table}[!htbp]
\centering
\marginnote{R-1.1, R-2.1}\caption{\textcolor{blue}{Positioning of the proposed sensor-depth PINN relative to representative physics-informed soil hydrology and control-oriented smart irrigation studies.}}
\label{tab:lit_comparison}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
{\color{blue}\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.165\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.175\textwidth}>{\raggedright\arraybackslash}p{0.185\textwidth}@{}}""",
    "Table 1 start",
)
text = tag_once(
    text,
    "\\textbf{Proposed PINN} & \\textbf{6 discrete sensor depths (5--55 cm)} & \\textbf{Flux balance + $c_1$ pos., $c_2$ mono., $c_3$ zero-flow} & \\textbf{Explicit $f_i$, $f_{i,i+1}$, $fr_i$, $fq_{\\mathrm{irr}}$, $fl_i$, $S_i$} & \\textbf{Two-stage RK4 ODE; control-admissible $q_{\\mathrm{irr}}, ET_0$} \\\\\n\\bottomrule\n\\end{tabular}",
    "\\textbf{Proposed PINN} & \\textbf{6 discrete sensor depths (5--55 cm)} & \\textbf{Flux balance + $c_1$ pos., $c_2$ mono., $c_3$ zero-flow} & \\textbf{Explicit $f_i$, $f_{i,i+1}$, $fr_i$, $fq_{\\mathrm{irr}}$, $fl_i$, $S_i$} & \\textbf{Two-stage RK4 ODE; control-admissible $q_{\\mathrm{irr}}, ET_0$} \\\\\n\\bottomrule\n\\end{tabular}}",
    "Table 1 end",
)
text = tag_once(
    text,
    r"We ask: Can PINNs that incorporate principles of Richards' equation and sensor data effectively model plant-soil-water dynamics to support predictive irrigation control and spatially localize crop water demands? Can such models be physically interpretable and useful for decision-making? We use Richards' equation as a structural framework, spatially discretizing it at sensor depths throughout the soil profile to obtain differential equations that describe the dynamics and relationships of each soil layer with irrigation ($q_{\mathrm{irr}}$), rainfall ($r$), and potential evapotranspiration ($ET_0$), as flow functions that satisfy water balance constraints. Evapotranspirative water depletion appears as a sink function of $ET_0$ and soil moisture at each layer. All flow functions are approximated using neural network branches that are regularized for physical consistency (non-negativity, monotonicity, and zero-forcing zero-flux). The PINN is trained in two stages: initially, it minimizes errors against measured soil moisture variations, followed by tuning through the incorporation of the fourth-order Runge--Kutta (RK4) numerical solution into the cost function. We validate this approach using field data from citrus production in Florida, where shallow sandy soils and saltwater intrusion create challenging conditions, contextualizing its 3-day (\SI{1.23}{\%VWC} RMSE) and rolling unseen-window performance, alongside physical control-admissibility audits, against persistence, linear state-space ODE, black-box Neural ODE, and ablated PINN baselines.",
    r"We ask: Can PINNs that incorporate principles of Richards' equation and sensor data effectively model plant-soil-water dynamics to \textcolor{blue}{support predictive irrigation control} and spatially localize crop water demands? Can such models be physically interpretable and useful for decision-making? We use Richards' equation as a structural framework, spatially discretizing it at sensor depths throughout the soil profile to obtain differential equations that describe the dynamics and relationships of each soil layer with irrigation \textcolor{blue}{($q_{\mathrm{irr}}$)}, rainfall \textcolor{blue}{($r$)}, and potential evapotranspiration ($ET_0$), as flow functions that satisfy water balance constraints. \textcolor{blue}{Evapotranspirative water depletion} appears as a sink function of $ET_0$ and soil moisture at each layer. All flow functions are approximated using neural network branches that are \textcolor{blue}{regularized} for physical consistency (non-negativity, monotonicity, and zero\textcolor{blue}{-forcing zero-flux}). The PINN is trained in two stages: initially, it minimizes errors against measured soil moisture variations, followed by tuning through the incorporation of the \textcolor{blue}{fourth-order Runge--Kutta (RK4)} numerical solution into the cost function. We validate this approach using field data from citrus production in Florida, where shallow sandy soils and saltwater intrusion create challenging conditions\marginnote{E-1.1, R-1.3d, R-2.10}\textcolor{blue}{, contextualizing its 3-day (\SI{1.23}{\%VWC} RMSE) and rolling unseen-window performance, alongside physical control-admissibility audits, against persistence, linear state-space ODE, black-box Neural ODE, and ablated PINN baselines.}",
    "Intro Move 5",
)

# 3. Section 1 - Results
text = tag_once(
    text,
    r"\caption{Evaluation of the proposed PINN for multi-depth soil moisture prediction, showing measured rainfall ($r$, \si{mm/h}), irrigation duration fraction ($q_{\mathrm{irr}}$, \si{h/h}), FAO-56 Penman--Monteith evapotranspiration ($ET_0$, \si{mm/h}), and measured versus closed-loop estimated soil moisture (\si{\%VWC}) across 5--55~\si{cm} over the 35-day training period (839~h) and held-out 3-day forecast (72~h, after a 25-h buffer), driven solely by $m(0)$ and observed ($r, q_{\mathrm{irr}}, ET_0$). Moisture heatmaps use a 3.5--10.2~\si{\%VWC} contrast range ($\approx 90\%$ empirical interval; 5.8th--92.1st training percentiles). Lower panels show inferred evapotranspirative depletion rates ($pc_i$, \si{\%VWC/h}).}",
    r"\marginnote{R-2.3, R-2.4, R-2.12}\caption{\textcolor{blue}{Evaluation of the proposed PINN for multi-depth soil moisture prediction, showing measured rainfall ($r$, \si{mm/h}), irrigation duration fraction ($q_{\mathrm{irr}}$, \si{h/h}), FAO-56 Penman--Monteith evapotranspiration ($ET_0$, \si{mm/h}), and measured versus closed-loop estimated soil moisture (\si{\%VWC}) across 5--55~\si{cm} over the 35-day training period (839~h) and held-out 3-day forecast (72~h, after a 25-h buffer), driven solely by $m(0)$ and observed ($r, q_{\mathrm{irr}}, ET_0$). Moisture heatmaps use a 3.5--10.2~\si{\%VWC} contrast range ($\approx 90\%$ empirical interval; 5.8th--92.1st training percentiles). Lower panels show inferred evapotranspirative depletion rates ($pc_i$, \si{\%VWC/h}).}}",
    "Fig 1 caption",
)
text = tag_once(
    text,
    r"The results of the evaluation of the proposed PINN approach are shown in Fig.~\ref{Mainfig}, where rain ($r$), irrigation ($q_{\mathrm{irr}}$), reference evapotranspiration ($ET_0$), and soil moisture data used during the 35-day training and held-out 3-day forecasting periods are presented. In the moisture heatmaps, the color range of 3.5--10.2~\si{\%VWC} spans the dominant $\approx 90\%$ empirical data range (specifically the 5.8th to 92.1st percentiles of the training dataset, enclosing 86.3\% of training observations) to preserve visual contrast across intermediate depths while preventing extreme transient storm spikes in topsoil from saturating the color scale. This visualization helps reveal soil-water dynamics not apparent in common one-dimensional plots. The sampling time used for data management, visualization, and modeling was one hour (\(\Delta t = \SI{1}{h}\)) to align with the availability of data from forecasting services \cite{NWS2024}. The heatmap shows soil moisture over time at depths of 5, 15, 25, 35, 45, and 55 \si{cm}; this spatial discretization is dictated by the number of soil moisture sensors available for data collection. The figure also presents the estimated evapotranspirative depletion rates ($pc_i$, denoted $S_i$ in Eq.~\eqref{system}) at the different depths.",
    r"The results of the evaluation of the proposed PINN approach are shown in Fig.~\ref{Mainfig}, where rain \textcolor{blue}{($r$)}, irrigation \textcolor{blue}{($q_{\mathrm{irr}}$), reference evapotranspiration ($ET_0$)}, and soil moisture data used during the \textcolor{blue}{35-day} training and \textcolor{blue}{held-out 3-day} forecasting periods are presented. \marginnote{R-2.12}\textcolor{blue}{In the moisture heatmaps, the color range of 3.5--10.2~\si{\%VWC} spans the dominant $\approx 90\%$ empirical data range (specifically the 5.8th to 92.1st percentiles of the training dataset, enclosing 86.3\% of training observations) to preserve visual contrast across intermediate depths while preventing extreme transient storm spikes in topsoil from saturating the color scale.} This visualization helps reveal soil-water dynamics not apparent in common one-dimensional plots. The sampling time used for data management, visualization, and modeling was one hour \textcolor{blue}{(\(\Delta t = \SI{1}{h}\))} to align with the availability of data from forecasting services \cite{NWS2024}. The heatmap shows soil moisture over time at depths of 5, 15, 25, 35, 45, and 55 \si{cm}; this spatial discretization is dictated by the number of soil moisture sensors available for data collection. The figure also presents the estimated \textcolor{blue}{evapotranspirative depletion rates ($pc_i$, denoted $S_i$ in Eq.~\eqref{system})} at the different depths.",
    "Results Para 1",
)
text = tag_once(
    text,
    r"In Fig.~\ref{Mainfig}, every predicted soil moisture trajectory in both training (839~h open rollout from a single initial condition) and held-out 3-day forecasting (72~h rollout from the held-out May~3 21:00:00 state) is generated purely in closed loop using only initial conditions and observed external inputs ($r$, $q_{\mathrm{irr}}$, $ET_0$) without state measurement assimilation, validating our proposed PINN as a control-oriented model,",
    r"In Fig.~\ref{Mainfig}, \marginnote{R-2.3, R-2.4}\textcolor{blue}{every predicted soil moisture trajectory in both training (839~h open rollout from a single initial condition) and held-out 3-day forecasting (72~h rollout from the held-out May~3 21:00:00 state) is generated purely in closed loop} using only initial conditions and \textcolor{blue}{observed external inputs ($r$, $q_{\mathrm{irr}}$,} $ET_0$\textcolor{blue}{) without state measurement assimilation}, validating our proposed PINN as a control-oriented model,",
    "Results Para 2",
)
text = tag_once(
    text,
    r"The model simultaneously estimates evapotranspirative water depletion ($pc_i$) at each depth as a function of soil moisture and $ET_0$. This estimation enables the development of more precise control strategies by helping localize depths where soil moisture must be guaranteed. In Fig.~\ref{Mainfig}, the estimated depletion rates from 5 to 55 \si{cm} depths are illustrated. The results show depletion patterns correlated with both soil moisture and $ET_0$ across all layers. The estimation is performed in units of \si{\%VWC/h} (numerically equivalent to \si{mm/h} per \SI{10}{cm} soil compartment), indicating how much soil moisture decreases over time in relation to soil moisture and $ET_0$. Pronounced maxima are estimated at the first layer (5 \si{cm}) and the lowest layer (55 \si{cm}), where significant portions of the depletion reflect non-transpirative processes: in the first layer, because it is directly exposed to surface evaporation, and in the lowest layer, because the crop is planted on a raised bed flanked by a drainage ditch whose water-table drawdown and bank evaporation vary in phase with $ET_0$ patterns. Within the upper root zone above the restrictive clay horizon, peak apparent root-zone depletion occurs at 25 \si{cm}. Physical and agronomic qualifications on interpreting $pc_i$ as combined evapotranspirative depletion rather than directly measured transpiration are detailed in Section~\ref{sec:uptake_rationale}.",
    r"The model simultaneously estimates \marginnote{R-2.6}\textcolor{blue}{evapotranspirative water depletion ($pc_i$)} at each depth as a function of soil moisture and $ET_0$. This estimation enables the development of more precise control strategies by helping localize depths where soil moisture must be guaranteed. In Fig.~\ref{Mainfig}, the estimated \textcolor{blue}{depletion rates} from 5 to 55 \si{cm} depths \textcolor{blue}{are} illustrated. The results show \textcolor{blue}{depletion} patterns correlated with both soil moisture and $ET_0$ across all layers. The estimation is performed in units of \si{\%VWC/h} \textcolor{blue}{(numerically equivalent to \si{mm/h} per \SI{10}{cm} soil compartment)}, indicating how much soil moisture decreases over time in relation to soil moisture and $ET_0$. \textcolor{blue}{Pronounced maxima} are estimated at the first layer (5 \si{cm}) and the lowest layer (55 \si{cm}), \textcolor{blue}{where significant portions of the depletion reflect non-transpirative processes: in the first layer,} because it is directly exposed to \textcolor{blue}{surface evaporation}, and in the lowest layer, \textcolor{blue}{because the crop is planted on a raised bed flanked by a drainage ditch whose water-table drawdown and bank evaporation vary in phase with} $ET_0$ patterns. \textcolor{blue}{Within the upper root zone above the restrictive clay horizon, peak apparent root-zone depletion occurs} at 25 \si{cm}\textcolor{blue}{. Physical and agronomic qualifications on interpreting $pc_i$ as combined evapotranspirative depletion rather than directly measured transpiration are detailed in Section~\ref{sec:uptake_rationale}.}",
    "Results Para 4 pc_i",
)
text = tag_once(
    text,
    r"\caption{Soil moisture prediction residuals ($m_i - \hat{m}_i$, \si{\%VWC}) across 5--55~\si{cm} during the 35-day training and 3-day held-out forecasting periods. Black frames delineate heatmap boundaries against zero-error white regions ($-2.5$ to $11.1$~\si{\%VWC}; red under-prediction, blue over-prediction). Bottom boxplots show near-zero medians and interquartile ranges below 2.5~\si{\%VWC} across all depths.}",
    r"\marginnote{R-1.8}\caption{\textcolor{blue}{Soil moisture prediction residuals ($m_i - \hat{m}_i$, \si{\%VWC}) across 5--55~\si{cm} during the 35-day training and 3-day held-out forecasting periods. Black frames delineate heatmap boundaries against zero-error white regions ($-2.5$ to $11.1$~\si{\%VWC}; red under-prediction, blue over-prediction). Bottom boxplots show near-zero medians and interquartile ranges below 2.5~\si{\%VWC} across all depths.}}",
    "Fig 2 caption",
)
text = tag_once(
    text,
    r"""\caption{Layer-wise and pooled statistical evaluation of the archived PINN checkpoint across the 35-day training rollout ($N = 839\,\mathrm{h} \times 6 = 5{,}034$) and 3-day held-out forecast ($N = 72\,\mathrm{h} \times 6 = 432$). $\sigma_m$ is measured VWC standard deviation, RMSE$^\dagger$ is in \si{\%VWC} (identically \si{mm} per \SI{10}{cm} layer), SD is residual standard deviation, $R^2 = 1 - SS_{\mathrm{res}}/SS_{\mathrm{tot}}$, and $r$ ($p$) is Pearson correlation with two-sided Student's $t$-test $p$-values.}
\label{tab:stats_combined}
\fontsize{6.2}{7.8}\selectfont
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}lcccccccccc@{}}""",
    r"""\marginnote{E-1.1, R-1.3a, R-2.12}\caption{\textcolor{blue}{Layer-wise and pooled statistical evaluation of the archived PINN checkpoint across the 35-day training rollout ($N = 839\,\mathrm{h} \times 6 = 5{,}034$) and 3-day held-out forecast ($N = 72\,\mathrm{h} \times 6 = 432$). $\sigma_m$ is measured VWC standard deviation, RMSE$^\dagger$ is in \si{\%VWC} (identically \si{mm} per \SI{10}{cm} layer), SD is residual standard deviation, $R^2 = 1 - SS_{\mathrm{res}}/SS_{\mathrm{tot}}$, and $r$ ($p$) is Pearson correlation with two-sided Student's $t$-test $p$-values.}}
\label{tab:stats_combined}
\fontsize{6.2}{7.8}\selectfont
\setlength{\tabcolsep}{1.8pt}
{\color{blue}\begin{tabular}{@{}lcccccccccc@{}}""",
    "Table 2 start",
)
text = tag_once(
    text,
    r"""Pooled & 2.53 & 1.16 & 1.12 & 0.796 & 0.901 ($<10^{-300}$) & 2.81 & 1.23 & 1.24 & 0.807 & 0.921 ($7.9\!\times\!10^{-178}$) \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item $^\dagger$\SI{1.0}{\%VWC} equals \SI{1.0}{mm} water depth per \SI{10}{cm} compartment; integrated \SI{60}{cm} profile RMSE is \SI{3.19}{mm} (training) and \SI{3.22}{mm} (forecasting). Metrics reflect exact evaluation of \texttt{pinn\_weights\_H56\_lr7.97e-05.weights.h5} (updating pre-archive summary values of 1.19 / 1.25~\si{\%VWC}). Two-sided $t$-test uses $t = r\sqrt{(N-2)/(1-r^2)}$ ($<10^{-300}$: IEEE~754 float64 underflow).
\end{tablenotes}""",
    r"""Pooled & 2.53 & 1.16 & 1.12 & 0.796 & 0.901 ($<10^{-300}$) & 2.81 & 1.23 & 1.24 & 0.807 & 0.921 ($7.9\!\times\!10^{-178}$) \\
\bottomrule
\end{tabular}}
\begin{tablenotes}
\footnotesize
\item \textcolor{blue}{$^\dagger$\SI{1.0}{\%VWC} equals \SI{1.0}{mm} water depth per \SI{10}{cm} compartment; integrated \SI{60}{cm} profile RMSE is \SI{3.19}{mm} (training) and \SI{3.22}{mm} (forecasting). Metrics reflect exact evaluation of \texttt{pinn\_weights\_H56\_lr7.97e-05.weights.h5} (updating pre-archive summary values of 1.19 / 1.25~\si{\%VWC}). Two-sided $t$-test uses $t = r\sqrt{(N-2)/(1-r^2)}$ ($<10^{-300}$: IEEE~754 float64 underflow).}
\end{tablenotes}""",
    "Table 2 end",
)
text = tag_once(
    text,
    r"Quantitative predictive metrics are reported in Table~\ref{tab:stats_combined}, Fig.~\ref{errorfig}, and the parity scatterplots in Fig.~\ref{fig:scatter_parity}. Over the held-out 3-day forecast, the closed-loop PINN achieves a pooled RMSE of \SI{1.23}{\%VWC} (SD = \SI{1.24}{\%VWC}, $R^2 = 0.807$, Pearson $r = 0.921$, $p = \num{7.94e-178}$), closely matching its 35-day training performance (RMSE = \SI{1.16}{\%VWC}, SD = \SI{1.12}{\%VWC}, $R^2 = 0.796$, $r = 0.901$, $p < 10^{-300}$). Translated to actionable water depth, each \SI{1}{\%VWC} error across a \SI{10}{cm} soil compartment represents \SI{1.0}{mm} of stored water; per-layer 3-day RMSE spans \SI{0.28}{mm} at 35~\si{cm} to \SI{2.02}{mm} at 5~\si{cm}, while integrated \SI{60}{cm} root-zone storage RMSE is \SI{3.22}{mm} (8.55\% of the \SI{37.70}{mm} mean profile storage) in forecasting and \SI{3.19}{mm} in training.",
    r"\marginnote{E-1.1, R-1.3a}\textcolor{blue}{Quantitative predictive metrics are reported in Table~\ref{tab:stats_combined}, Fig.~\ref{errorfig}, and the parity scatterplots in Fig.~\ref{fig:scatter_parity}. Over the held-out 3-day forecast, the closed-loop PINN achieves a pooled RMSE of \SI{1.23}{\%VWC} (SD = \SI{1.24}{\%VWC}, $R^2 = 0.807$, Pearson $r = 0.921$, $p = \num{7.94e-178}$), closely matching its 35-day training performance (RMSE = \SI{1.16}{\%VWC}, SD = \SI{1.12}{\%VWC}, $R^2 = 0.796$, $r = 0.901$, $p < 10^{-300}$). Translated to actionable water depth, each \SI{1}{\%VWC} error across a \SI{10}{cm} soil compartment represents \SI{1.0}{mm} of stored water; per-layer 3-day RMSE spans \SI{0.28}{mm} at 35~\si{cm} to \SI{2.02}{mm} at 5~\si{cm}, while integrated \SI{60}{cm} root-zone storage RMSE is \SI{3.22}{mm} (8.55\% of the \SI{37.70}{mm} mean profile storage) in forecasting and \SI{3.19}{mm} in training.}",
    "Results mm depth",
)
text = tag_once(
    text,
    r"Table~\ref{tab:stats_combined} and Fig.~\ref{fig:scatter_parity} also clarify two statistical subtleties regarding concatenation and low-variance subsoils. First, the ``Pooled'' row concatenates all six time-aligned depth vectors into a single vector of length $6 \times 839 = 5{,}034$ (training) and $6 \times 72 = 432$ (forecasting), computing $SS_{\mathrm{res}} = \sum_{t,i} (m_i(t) - \hat{m}_i(t))^2$ against global profile variance $SS_{\mathrm{tot}} = \sum_{t,i} (m_i(t) - \bar{m})^2$ ($\sigma_{\mathrm{pooled}} = 2.81\,\si{\%VWC}$, $SS_{\mathrm{tot}} = 3{,}406.8$); pooled $R^2 = 0.807$ thus captures vertical stratification and storm dynamics across the column. Second, within individual deep layers during the 3-day window—25~\si{cm} ($\sigma_m = 0.87\,\si{\%VWC}$), 45~\si{cm} ($\sigma_m = 0.34\,\si{\%VWC}$), and 55~\si{cm} ($\sigma_m = 0.17\,\si{\%VWC}$)—measured moisture is nearly static. Because single-layer $R^2_i = 1 - \mathrm{MSE}_i/\mathrm{Var}(m_i)$ divides squared error by near-zero variance at 45 and 55~\si{cm} ($\mathrm{Var} \approx 0.11$ and $0.03\,\si{\%VWC}^2$), small offset biases of \SI{0.95}{mm} and \SI{1.32}{mm} yield strongly negative $R^2_i$ ($-7.043$ and $-59.517$) despite significant positive correlations ($r = 0.709$ and $0.713$, $p \le \num{3.2e-12}$). Consequently, pooled $R^2$ must be interpreted alongside per-layer millimeter RMSE and Pearson $r$, as it does not imply high explained variance on quasi-static deep horizons.",
    r"\marginnote{R-1.3b, R-1.3c, R-2.5}\textcolor{blue}{Table~\ref{tab:stats_combined} and Fig.~\ref{fig:scatter_parity} also clarify two statistical subtleties regarding concatenation and low-variance subsoils. First, the ``Pooled'' row concatenates all six time-aligned depth vectors into a single vector of length $6 \times 839 = 5{,}034$ (training) and $6 \times 72 = 432$ (forecasting), computing $SS_{\mathrm{res}} = \sum_{t,i} (m_i(t) - \hat{m}_i(t))^2$ against global profile variance $SS_{\mathrm{tot}} = \sum_{t,i} (m_i(t) - \bar{m})^2$ ($\sigma_{\mathrm{pooled}} = 2.81\,\si{\%VWC}$, $SS_{\mathrm{tot}} = 3{,}406.8$); pooled $R^2 = 0.807$ thus captures vertical stratification and storm dynamics across the column. Second, within individual deep layers during the 3-day window—25~\si{cm} ($\sigma_m = 0.87\,\si{\%VWC}$), 45~\si{cm} ($\sigma_m = 0.34\,\si{\%VWC}$), and 55~\si{cm} ($\sigma_m = 0.17\,\si{\%VWC}$)—measured moisture is nearly static. Because single-layer $R^2_i = 1 - \mathrm{MSE}_i/\mathrm{Var}(m_i)$ divides squared error by near-zero variance at 45 and 55~\si{cm} ($\mathrm{Var} \approx 0.11$ and $0.03\,\si{\%VWC}^2$), small offset biases of \SI{0.95}{mm} and \SI{1.32}{mm} yield strongly negative $R^2_i$ ($-7.043$ and $-59.517$) despite significant positive correlations ($r = 0.709$ and $0.713$, $p \le \num{3.2e-12}$). Consequently, pooled $R^2$ must be interpreted alongside per-layer millimeter RMSE and Pearson $r$, as it does not imply high explained variance on quasi-static deep horizons.}",
    "Results R2 variance",
)
text = tag_once(
    text,
    r"\caption{Predicted versus measured soil moisture parity scatterplots across all six sensor depths (5--55~\si{cm}) for the 35-day training period ($N = 5{,}034$ concatenated depth--hour points) and the 3-day held-out forecast ($N = 432$ points), illustrating tight clustering around the 1:1 line for low-variance deep layers (45--55~\si{cm}) alongside the profile-wide vertical span underlying the pooled $R^2 = 0.807$ and Pearson $r = 0.921$.}",
    r"\marginnote{R-1.3c}\caption{\textcolor{blue}{Predicted versus measured soil moisture parity scatterplots across all six sensor depths (5--55~\si{cm}) for the 35-day training period ($N = 5{,}034$ concatenated depth--hour points) and the 3-day held-out forecast ($N = 432$ points), illustrating tight clustering around the 1:1 line for low-variance deep layers (45--55~\si{cm}) alongside the profile-wide vertical span underlying the pooled $R^2 = 0.807$ and Pearson $r = 0.921$.}}",
    "Fig 3 scatter caption",
)
text = tag_once(
    text,
    r"In Fig.~\ref{errorfig}, the estimation error is distributed around zero during training and forecasting, where the box plots show that most of the error is distributed around zero (with interquartile ranges below \SI{2.5}{\%VWC}), along with some outliers due to the conservative behavior of the proposed PINN.",
    r"In \textcolor{blue}{Fig.~\ref{errorfig}}, the estimation error is distributed around zero during training and forecasting\textcolor{blue}{, where the box plots show that most of the error is distributed around zero (with interquartile ranges below \SI{2.5}{\%VWC}), along with some outliers due to the conservative behavior of the proposed PINN}.",
    "Results Fig 2 prose",
)
text = tag_once(
    text,
    r"\caption{Estimated flux branches describing soil-water dynamics across sensor depths (5--55~\si{cm}, labeled along the left margin): $fr_i(r)$ is direct rainfall infiltration (\si{\%VWC/h}); $f_i(m_i)$ is gravitational vertical outflow (\si{\%VWC/h}); $f_{i,i+1}(m_i,m_{i+1})$ is signed potential-driven inter-layer exchange (\si{\%VWC/h}); $S_i(m_i,ET_0)$ is evapotranspirative depletion (\si{\%VWC/h}); $fl_i(m_i)$ is lateral ditch outflow (\si{\%VWC/h}, with scientific y-axis notation for compactness); and $fq_{\mathrm{irr}}(q_{\mathrm{irr}})$ is surface sprinkler inflow (\si{\%VWC/h}). Light blue regions denote the measured training moisture bounds at each depth.}",
    r"\marginnote{R-1.7, R-2.12}\caption{\textcolor{blue}{Estimated flux branches describing soil-water dynamics across sensor depths (5--55~\si{cm}, labeled along the left margin): $fr_i(r)$ is direct rainfall infiltration (\si{\%VWC/h}); $f_i(m_i)$ is gravitational vertical outflow (\si{\%VWC/h}); $f_{i,i+1}(m_i,m_{i+1})$ is signed potential-driven inter-layer exchange (\si{\%VWC/h}); $S_i(m_i,ET_0)$ is evapotranspirative depletion (\si{\%VWC/h}); $fl_i(m_i)$ is lateral ditch outflow (\si{\%VWC/h}, with scientific y-axis notation for compactness); and $fq_{\mathrm{irr}}(q_{\mathrm{irr}})$ is surface sprinkler inflow (\si{\%VWC/h}). Light blue regions denote the measured training moisture bounds at each depth.}}",
    "Fig 4 caption",
)
text = tag_once(
    text,
    r"""Furthermore, the estimated water movement associated with the potential difference between two soil layers is interlayer flow ($f_{i,i+1}$), ranging from $-6$ \si{\%VWC/h} (dark blue, upstream) to 3 \si{\%VWC/h} (yellow, downstream). It is noted that flows are generally consistent with potential gradients but are often negligible, consistent with sandy soil hydraulic properties \cite{Reyes2016}. The observed hydraulic trends between layers 1 and 2 can be attributed to changes in the soil induced by organic matter and agricultural practices.
The estimated evapotranspirative depletion ($S_i$, or $pc_i$) as a function of soil moisture and $ET_0$ increases with both. It is highlighted that layer 3 (25 \si{cm}) shows the maximum potential within the upper root zone to produce high moisture extraction, where at maxima of $ET_0$ and soil moisture, the produced change in soil moisture can be around 0.8 \si{\%VWC/h}, corroborating the proposed PINN as a tool to estimate soil regions where water must be guaranteed.
The estimated lateral outflow ($fl_i$) does not appear to be relevant for direct comparisons with the vertical movement of water. A particular small negative, non-monotonic behavior (despite the included constraints) appears in layer 5 ($fl_5(m_5)$) and reflects both lateral inflows from an adjacent ditch during storm surges and the fact that penalty terms $c_1$--$c_3$ act as finite soft regularizers rather than hard algebraic barriers.
Finally, it is estimated how irrigation ($fq_{\mathrm{irr}}(q_{\mathrm{irr}})$) increases the soil moisture. In this case, irrigation is considered in the surface layer because the field used sprinkler irrigation; however, the PINN configuration can be adjusted to the selected irrigation system by setting this inflow to the layer where it is applied. The estimated function is predominantly linear, where soil moisture in the surface layer increases by approximately 5 \si{\%VWC/h} per hour of irrigation. This is crucial for control and decision-making processes. For instance, this curve enables us to quickly determine the irrigation required to bring a crop from a measured soil moisture level to a desired point that ensures adequate water uptake in the surface layers, which is crucial during the early stages of perennial crops.
The shaded blue regions highlight the span of measured soil moisture data used for training, making evident the benefits and effects of the soft physical regularizers in the formulated approach, where, from sparse data, some functions like the vertical flow and lateral flow are reconstructed from the origin to far values following the positivity, monotonicity, and zero-forcing zero-flow regularizers.""",
    r"""Furthermore, the estimated water movement associated with the potential difference between two soil layers is interlayer flow \textcolor{blue}{($f_{i,i+1}$)}, ranging from \textcolor{blue}{$-6$ \si{\%VWC/h}} (dark blue, upstream) to 3 \si{\%VWC/h} (yellow, downstream). It is noted that flows are generally consistent with potential gradients but are often negligible, consistent with sandy soil hydraulic properties \cite{Reyes2016}. The observed hydraulic trends between layers 1 and 2 can be attributed to changes in the soil induced by organic matter and agricultural practices.
The estimated \marginnote{R-2.6}\textcolor{blue}{evapotranspirative depletion ($S_i$, or $pc_i$)} as a function of soil moisture and $ET_0$ increases with both. It is highlighted that layer 3 \textcolor{blue}{(25 \si{cm})} shows the maximum potential \textcolor{blue}{within the upper root zone} to produce high \textcolor{blue}{moisture extraction}, where at maxima of $ET_0$ and soil moisture, the produced change in soil moisture can be around 0.8 \si{\%VWC/h}, corroborating the proposed PINN as a tool to estimate soil regions where water must be guaranteed.
The estimated lateral outflow \textcolor{blue}{($fl_i$)} does not appear to be relevant for direct comparisons with the vertical movement of water. A particular small negative, non-monotonic behavior (despite the included constraints) appears in layer 5 \marginnote{R-2.8}\textcolor{blue}{($fl_5(m_5)$) and reflects both} lateral inflows from an adjacent ditch \textcolor{blue}{during storm surges and the fact that penalty terms $c_1$--$c_3$ act as finite soft regularizers rather than hard algebraic barriers}.
Finally, it is estimated how irrigation \textcolor{blue}{($fq_{\mathrm{irr}}(q_{\mathrm{irr}})$)} increases the soil moisture. In this case, irrigation is considered in the surface layer because the field used sprinkler irrigation; however, the PINN configuration can be adjusted to the selected irrigation system by setting this inflow to the layer where it is applied. The estimated function is predominantly linear, where soil moisture in the surface layer increases by approximately 5 \si{\%VWC/h} per hour of irrigation. This is crucial for control and decision-making processes. For instance, this curve enables us to quickly determine the irrigation required to bring a crop from a measured soil moisture level to a desired point that ensures adequate water uptake in the surface layers, which is crucial during the early stages of perennial crops.
The shaded blue regions highlight the span of measured soil moisture data used for training, making evident the benefits and effects of the \textcolor{blue}{soft physical regularizers} in the formulated approach, where, from sparse data, some functions like the vertical flow and lateral flow are reconstructed from the origin to far values following the positivity, monotonicity, and zero\textcolor{blue}{-forcing zero-flow regularizers.}""",
    "Results Fig 4 prose",
)

# New Section 1.1 & 1.2 + Conclusions
text = tag_once(
    text,
    r"\subsection{Benchmark Comparison, Ablation Study, and Control-Oriented Admissibility}\label{sec:benchmarks}",
    r"{\color{blue}\subsection{Benchmark Comparison, Ablation Study, and Control-Oriented Admissibility}\label{sec:benchmarks}\marginnote{R-1.3d, R-2.10}",
    "Benchmarks start",
)
text = tag_once(
    text,
    r"\subsection{Agronomic and Hydrological Interpretation of Inferred Evapotranspirative Depletion}\label{sec:uptake_rationale}",
    r"\subsection{Agronomic and Hydrological Interpretation of Inferred Evapotranspirative Depletion}\label{sec:uptake_rationale}\marginnote{R-1.3d, R-2.6}",
    "Uptake tag",
)
text = tag_once(
    text,
    r"Partitioning physiological transpiration from soil evaporation and ditch exchange would require independent sap-flow or lysimetry sensors beyond the scope of this study.",
    r"Partitioning physiological transpiration from soil evaporation and ditch exchange would require independent sap-flow or lysimetry sensors beyond the scope of this study.}",
    "Uptake end",
)
text = tag_once(
    text,
    r"\subsection{Conclusions}",
    r"{\color{blue}\subsection{Conclusions}\marginnote{E-1.1, R-2.11}",
    "Conclusions start",
)
text = tag_once(
    text,
    r"closed-loop field MPC deployment and direct measurement of irrigation water savings remain essential future work rather than demonstrated outcomes of this study.",
    r"closed-loop field MPC deployment and direct measurement of irrigation water savings remain essential future work rather than demonstrated outcomes of this study.}",
    "Conclusions end",
)

# 4. Section 2 - Methodology
text = tag_once(
    text,
    r"""Therefore, the model must accurately describe the vertical dynamics of soil moisture and their interaction with evapotranspirative plant-soil sink processes, environmental variables, and irrigation. Specifically, the model must describe the soil moisture dynamics at the targeted measured soil layers $m_i(t)$ (\si{\%VWC}). 
Moreover, at each layer, the model must explain the effects of measurable external inputs, such as rainfall $r(t)$ (\si{mm/h}), irrigation rate $q_{\mathrm{irr}}(t)$ (\si{h/h}), and potential evapotranspiration $ET_0(t)$ (\si{mm/h}), which drives root-zone water depletion.""",
    r"""Therefore, the model must accurately describe the vertical dynamics of soil moisture and their interaction with \marginnote{R-1.2, R-2.6}\textcolor{blue}{evapotranspirative plant-soil sink processes}, environmental variables, and irrigation. Specifically, the model must describe the soil moisture dynamics at the targeted measured soil layers $m_i(t)$ \textcolor{blue}{(\si{\%VWC})}. 
Moreover, at each layer, the model must explain the effects of measurable external inputs, such as rainfall $r(t)$ (\si{mm/h}), irrigation rate \textcolor{blue}{$q_{\mathrm{irr}}(t)$} (\si{h/h}), and potential evapotranspiration \textcolor{blue}{$ET_0(t)$} (\si{mm/h}), which \textcolor{blue}{drives root-zone water depletion}.""",
    "Problem Statement",
)
text = tag_once(
    text,
    r"\caption{Architecture and two-stage training workflow of the proposed PINN. Inputs comprise the 6-layer moisture vector $\hat{m}(t)$ (5--55~\si{cm}), irrigation ($q_{\mathrm{irr}}$), rainfall ($r$), and reference evapotranspiration ($ET_0$). Inside $g(\hat{m},r,ET_0,q_{\mathrm{irr}},\boldsymbol{\Theta})$, neural branches model vertical drainage $f_i(m_i)$, signed inter-layer exchange $f_{i,i+1}(m_i,m_{i+1})$ ($i=1\dots5$), lateral ditch outflow $fl_i(m_i)$ ($i=1\dots5$; $f_6$ captures lower boundary discharge), rainfall infiltration $fr_i(r)$, surface irrigation $fq_{\mathrm{irr}}(q_{\mathrm{irr}})$, and evapotranspirative sink $S_i(m_i,ET_0)$ ($pc_i$). Stage~1 minimizes derivative loss $j_1$ and Stage~2 refines $\boldsymbol{\Theta}$ via RK4 trajectory tracking loss $j_2$, subject to soft penalties $c_1$--$c_3$.}",
    r"\marginnote{R-1.6, R-2.12}\caption{\textcolor{blue}{Architecture and two-stage training workflow of the proposed PINN. Inputs comprise the 6-layer moisture vector $\hat{m}(t)$ (5--55~\si{cm}), irrigation ($q_{\mathrm{irr}}$), rainfall ($r$), and reference evapotranspiration ($ET_0$). Inside $g(\hat{m},r,ET_0,q_{\mathrm{irr}},\boldsymbol{\Theta})$, neural branches model vertical drainage $f_i(m_i)$, signed inter-layer exchange $f_{i,i+1}(m_i,m_{i+1})$ ($i=1\dots5$), lateral ditch outflow $fl_i(m_i)$ ($i=1\dots5$; $f_6$ captures lower boundary discharge), rainfall infiltration $fr_i(r)$, surface irrigation $fq_{\mathrm{irr}}(q_{\mathrm{irr}})$, and evapotranspirative sink $S_i(m_i,ET_0)$ ($pc_i$). Stage~1 minimizes derivative loss $j_1$ and Stage~2 refines $\boldsymbol{\Theta}$ via RK4 trajectory tracking loss $j_2$, subject to soft penalties $c_1$--$c_3$.}}",
    "Fig 5 PINNsmb caption",
)
text = tag_once(
    text,
    r"Let $z$ denote the vertical depth coordinate taken positive downward, and let $H = h(m) - z$ denote total hydraulic head, where $h(m)$ is the soil-water matric pressure head and $-z$ is gravitational potential relative to the soil surface.",
    r"\marginnote{R-2.2}{\color{blue}Let $z$ denote the vertical depth coordinate taken positive downward, and let $H = h(m) - z$ denote total hydraulic head, where $h(m)$ is the soil-water matric pressure head and $-z$ is gravitational potential relative to the soil surface.",
    "Darcy sign start",
)
text = tag_once(
    text,
    r"where $f_i(m_i) \ge 0$ captures downward gravitational drainage and $f_{i,i+1}(m_i, m_{i+1})$ captures potential-driven inter-layer redistribution, which is intentionally left signed (positive downward when $m_i > m_{i+1}$, negative for upward capillary rise when $m_{i+1} > m_i$).",
    r"where $f_i(m_i) \ge 0$ captures downward gravitational drainage and $f_{i,i+1}(m_i, m_{i+1})$ captures potential-driven inter-layer redistribution, which is intentionally left signed (positive downward when $m_i > m_{i+1}$, negative for upward capillary rise when $m_{i+1} > m_i$).\marginnote{R-2.8}}",
    "Signed f_i,i+1 end",
)
text = tag_once(
    text,
    r"Specifically, $q_{\mathrm{irr}}$ is incorporated as the irrigation-related variable, and $fq_{\mathrm{irr}}(q_{\mathrm{irr}})$ is a function that explains the rate of change of the soil moisture in relation to irrigation. Note that $q_{\mathrm{irr}}$ can be expressed either in volumetric flow units (e.g., \si{m^3/h}) or in relative irrigation-duration units (\si{h/h}), indicating the fraction of the sampling hour during which irrigation was active (used in this field study). In the first case, $fq_{\mathrm{irr}}(q_{\mathrm{irr}})$ provides a relation between applied flow and change in soil moisture, and in the second case, $fq_{\mathrm{irr}}(q_{\mathrm{irr}})$ provides a relation between duration of irrigation time and change in soil moisture.",
    r"Specifically, \textcolor{blue}{$q_{\mathrm{irr}}$} is incorporated as the \textcolor{blue}{irrigation-related} variable, and \textcolor{blue}{$fq_{\mathrm{irr}}(q_{\mathrm{irr}})$} is a function that explains the rate of change of the soil moisture in relation to irrigation\textcolor{blue}{.} \marginnote{R-1.5, R-1.10}\textcolor{blue}{Note that $q_{\mathrm{irr}}$ can be expressed either in volumetric flow units (e.g., \si{m^3/h}) or in relative irrigation-duration units (\si{h/h}), indicating the fraction of the sampling hour during which irrigation was active (used in this field study).} In the first case, \textcolor{blue}{$fq_{\mathrm{irr}}(q_{\mathrm{irr}})$} provides a relation between applied flow and change in soil moisture, and in the second case, \textcolor{blue}{$fq_{\mathrm{irr}}(q_{\mathrm{irr}})$} provides a relation between duration of irrigation time and change in soil moisture.",
    "Model q_irr prose",
)
text = tag_once(
    text,
    r"""Furthermore, we incorporate a sink component to estimate soil moisture changes due to evapotranspirative depletion in each layer, which is a key motivation for this study. This sink is represented by an evapotranspirative sink function, $S_i (m_i, ET_0)$ (denoted $pc_i$ in Section~\ref{Results}), describing moisture depletion at each layer governed by atmospheric, plant, and soil conditions \cite{swap2024}. Specifically, we propose that this sink is a function of the available water at the respective layer ($m_i$) and the potential evapotranspiration ($ET_0$).
Additionally, we include an outlet lateral flow, $fl_i (m_i)$, associated with subsurface drainage toward adjacent ditches when excess water accumulates across layers. This is considered an outlet, as most crops are planted in raised beds between furrows or ditches to facilitate drainage.

In summary, we are formulating a PINN to estimate the dynamics of a system with $n$ soil layers ($n=6$) as follows:
\begin{equation}
\label{system}
\begin{aligned}
\frac{d{m}_1}{dt} = & - f_1({m}_1) - f_{1,2}({m}_1,{m}_2) - fl_1(m_1) - S_1({m}_1, ET_0) + fr_1(r) + fq_{\mathrm{irr}}(q_{\mathrm{irr}})\\
\frac{d{m}_i}{dt} = & f_{i-1}({m}_{i-1}) + f_{i-1,i}({m}_{i-1},{m}_i) - f_i({m}_i)- f_{i,i+1}({m}_i,{m}_{i+1}) - fl_i(m_i) \\ 
 & - S_i({m}_i, ET_0) + fr_i(r), \qquad i = 2,\dots,n-1,\\
\frac{d{m}_n}{dt} = & f_{n-1}({m}_{n-1}) + f_{n-1,n}({m}_{n-1},{m}_n) - f_n({m}_n) - S_n({m}_n, ET_0) + fr_n(r).
\end{aligned}
\end{equation}
Note that at layer $n$ ($n=6$, \SI{55}{cm}), we exclude the lateral flow term $fl_n$.""",
    r"""Furthermore, we incorporate a sink component to estimate soil moisture changes due to \marginnote{R-2.6}\textcolor{blue}{evapotranspirative depletion} in each layer, which is a key motivation for this study. This sink is represented by \textcolor{blue}{an evapotranspirative sink function}, $S_i (m_i, ET_0)$ \textcolor{blue}{(denoted $pc_i$ in Section~\ref{Results}), describing moisture depletion at each layer governed} by atmospheric, plant, and soil conditions \cite{swap2024}. Specifically, we propose that this \textcolor{blue}{sink} is a function of the available water at the respective layer ($m_i$) and the potential evapotranspiration ($ET_0$).
Additionally, we include an outlet lateral flow, \textcolor{blue}{$fl_i (m_i)$}, associated with \marginnote{R-1.4}\textcolor{blue}{subsurface drainage toward adjacent ditches when excess water accumulates across} layers. This is considered an outlet, as most crops are planted in raised beds between furrows \textcolor{blue}{or ditches} to facilitate drainage.

In summary, we are formulating a PINN to estimate the dynamics of a system with $n$ soil layers \textcolor{blue}{($n=6$)} as follows:
{\color{blue}\begin{equation}
\label{system}
\begin{aligned}
\frac{d{m}_1}{dt} = & - f_1({m}_1) - f_{1,2}({m}_1,{m}_2) - fl_1(m_1) - S_1({m}_1, ET_0) + fr_1(r) + fq_{\mathrm{irr}}(q_{\mathrm{irr}})\\
\frac{d{m}_i}{dt} = & f_{i-1}({m}_{i-1}) + f_{i-1,i}({m}_{i-1},{m}_i) - f_i({m}_i)- f_{i,i+1}({m}_i,{m}_{i+1}) - fl_i(m_i) \\ 
 & - S_i({m}_i, ET_0) + fr_i(r), \qquad i = 2,\dots,n-1,\\
\frac{d{m}_n}{dt} = & f_{n-1}({m}_{n-1}) + f_{n-1,n}({m}_{n-1},{m}_n) - f_n({m}_n) - S_n({m}_n, ET_0) + fr_n(r).
\end{aligned}
\end{equation}}
Note that at layer $n$ \textcolor{blue}{($n=6$, \SI{55}{cm})}, we exclude the lateral flow term $fl_n$.""",
    "Model sink fl_i system",
)
text = tag_once(
    text,
    r"we are proposing an NN with branches that emulate the functions \( g \in \{f_i, f_{i,i+1}, S_i, fq_{\mathrm{irr}}, fr_i, fl_i\} \). Therefore, each branch is a Multilayer Perceptron (MLP) with depth \( L \) and hidden width \( H_{\mathrm{b}} \), given by",
    r"we are proposing an NN with branches that emulate the functions \( g \in \{f_i, f_{i,i+1}, S_i, \textcolor{blue}{fq_{\mathrm{irr}}}, fr_i, fl_i\} \). Therefore, each branch is a Multilayer Perceptron (MLP) with depth \( L \) and \textcolor{blue}{hidden width \( H_{\mathrm{b}} \)}, given by",
    "PINN structure intro",
)
text = tag_once(
    text,
    r"""where \(\mathcal{A}(\cdot)\) is the activation function, and \(\displaystyle \boldsymbol{\Theta} = \{\mathbf{W}^{(\ell)},\,\mathbf{b}^{(\ell)}\}_{\ell=1}^L\) is the set of trainable parameters across branch depth \(L\) and hidden width \(H_{\mathrm{b}}\). Specifically, in the implemented architecture (\texttt{pinn\_model.py}): (i) non-negative single-input branches (\(f_1\dots f_5\), \(fq_{\mathrm{irr}}\), \(fr_1\dots fr_6\), and \(fl_1\dots fl_5\)) use \texttt{mish} activations with zero bias (\(\mathbf{b}^{(\ell)}=\mathbf{0}\)) and hidden width \(H_{\mathrm{b}}=H=56\) across three hidden layers (\(L=4\)); (ii) signed inter-layer exchange branches (\(f_{1,2}\dots f_{5,6}\)) and the lower boundary branch (\(f_6\)) use \texttt{tanh} activations (\(H_{\mathrm{b}}=H=56\), \(L=4\); zero bias on \(f_{i,i+1}\) and trainable bias on \(f_6\)) to accommodate bidirectional signed fluxes; and (iii) two-input evapotranspirative sink branches (\(S_1\dots S_6\)) use \texttt{mish} activations with zero bias and doubled hidden width \(H_{\mathrm{b}}=2H=112\) across four hidden layers (\(L=5\)). The input \(\mathbf{x}\) denotes the argument of each flux function.
In cases where the function depends on a single variable $\left(f_i(m_i),\;fq_{\mathrm{irr}}(q_{\mathrm{irr}}),\;fr_i(r),\;fl_i(m_i)\right)$, we have \(\mathbf{x}\in\mathbb{R}\), and when the function depends on two variables \(f_{i,i+1}(m_i,m_{i+1})\), \(\displaystyle \mathbf{x}=[m_i,m_{i+1}]^\top\in\mathbb{R}^2\); and for \(S_i(m_i,ET_0)\), \(\displaystyle \mathbf{x}=[m_i,ET_0]^\top\in\mathbb{R}^2\). Moreover, the dimensions of the first-layer weight matrices are \(\mathbf{W}^{(1)}\in\mathbb{R}^{H_{\mathrm{b}}\times1}\) for scalar inputs, and \(\mathbf{W}^{(1)}\in\mathbb{R}^{H_{\mathrm{b}}\times2}\) for two-dimensional inputs as encountered in \(f_{i,i+1}\) and \(S_i\).
\( \mathbf{W}^{(\ell)} \in \mathbb{R}^{H_{\mathrm{b}} \times H_{\mathrm{b}}} \) for \( \ell = 2, \dots, L-1 \), and \( \mathbf{W}^{(L)} \in \mathbb{R}^{1 \times H_{\mathrm{b}}} \). With biases \( \mathbf{b}^{(\ell)} \in \mathbb{R}^{H_{\mathrm{b}}} \) for \( \ell = 1, \dots, L-1 \), and \( \mathbf{b}^{(L)} \in \mathbb{R} \) (set to zero on all zero-origin branches and trainable on \(f_6\)), yielding an exact total of 375,089 trainable parameters across all 29 branches at \(H=56\).

Note that for brevity, a compact description of the system can be formulated as 

\begin{equation}
    \label{systemtheta}
    \dot{m}(t) = g(m,r,ET_0,q_{\mathrm{irr}},\boldsymbol{\Theta}).
\end{equation}


\subsubsection{Optimization Problem and Training}\label{subsec2}

As shown in the model formulation, PINN is a highly flexible technique that allows incorporating all physical and even empirical knowledge about the system into the PINN structure. Additionally, the formulation of the optimization problem provides an opportunity to shape the estimated system, including information that can contribute to model generalization and the estimated system's physical correctness. Therefore, the optimization problem is formulated to parameterize the system~\eqref{systemtheta}, aiming that this: i) mimics soil-water and evapotranspirative water depletion dynamics;""",
    r"""where \(\mathcal{A}(\cdot)\) is the activation function, and \(\displaystyle \boldsymbol{\Theta} = \{\mathbf{W}^{(\ell)},\,\mathbf{b}^{(\ell)}\}_{\ell=1}^L\) is the set of trainable parameters \marginnote{R-2.8, R-2.9}\textcolor{blue}{across branch depth \(L\) and hidden width \(H_{\mathrm{b}}\). Specifically, in the implemented architecture (\texttt{pinn\_model.py}): (i) non-negative single-input branches (\(f_1\dots f_5\), \(fq_{\mathrm{irr}}\), \(fr_1\dots fr_6\), and \(fl_1\dots fl_5\)) use \texttt{mish} activations with zero bias (\(\mathbf{b}^{(\ell)}=\mathbf{0}\)) and hidden width \(H_{\mathrm{b}}=H=56\) across three hidden layers (\(L=4\)); (ii) signed inter-layer exchange branches (\(f_{1,2}\dots f_{5,6}\)) and the lower boundary branch (\(f_6\)) use \texttt{tanh} activations (\(H_{\mathrm{b}}=H=56\), \(L=4\); zero bias on \(f_{i,i+1}\) and trainable bias on \(f_6\)) to accommodate bidirectional signed fluxes; and (iii) two-input evapotranspirative sink branches (\(S_1\dots S_6\)) use \texttt{mish} activations with zero bias and doubled hidden width \(H_{\mathrm{b}}=2H=112\) across four hidden layers (\(L=5\)).} The input \(\mathbf{x}\) denotes the argument of each flux function.
In cases where the function depends on a single variable \textcolor{blue}{$\left(f_i(m_i),\;fq_{\mathrm{irr}}(q_{\mathrm{irr}}),\;fr_i(r),\;fl_i(m_i)\right)$}, we have \(\mathbf{x}\in\mathbb{R}\), and when the function depends on two variables \(f_{i,i+1}(m_i,m_{i+1})\), \textcolor{blue}{\(\displaystyle \mathbf{x}=[m_i,m_{i+1}]^\top\in\mathbb{R}^2\)}; and for \textcolor{blue}{\(S_i(m_i,ET_0)\)}, \textcolor{blue}{\(\displaystyle \mathbf{x}=[m_i,ET_0]^\top\in\mathbb{R}^2\)}. Moreover, the dimensions of the first-layer weight matrices are \textcolor{blue}{\(\mathbf{W}^{(1)}\in\mathbb{R}^{H_{\mathrm{b}}\times1}\)} for scalar inputs, and \textcolor{blue}{\(\mathbf{W}^{(1)}\in\mathbb{R}^{H_{\mathrm{b}}\times2}\)} for two-dimensional inputs as encountered in \(f_{i,i+1}\) and \(S_i\).
\textcolor{blue}{\( \mathbf{W}^{(\ell)} \in \mathbb{R}^{H_{\mathrm{b}} \times H_{\mathrm{b}}} \)} for \( \ell = 2, \dots, L-1 \), and \textcolor{blue}{\( \mathbf{W}^{(L)} \in \mathbb{R}^{1 \times H_{\mathrm{b}}} \)}. With biases \textcolor{blue}{\( \mathbf{b}^{(\ell)} \in \mathbb{R}^{H_{\mathrm{b}}} \)} for \( \ell = 1, \dots, L-1 \), and \textcolor{blue}{\( \mathbf{b}^{(L)} \in \mathbb{R} \) (set to zero on all zero-origin branches and trainable on \(f_6\)), yielding an exact total of 375,089 trainable parameters across all 29 branches at \(H=56\).}

Note that for brevity, a compact description of the system can be formulated as 

\begin{equation}
    \label{systemtheta}
    \dot{m}(t) = g(m,r,ET_0,\textcolor{blue}{q_{\mathrm{irr}}},\boldsymbol{\Theta})\textcolor{blue}{.}
\end{equation}


\subsubsection{Optimization Problem and Training}\label{subsec2}

As shown in the model formulation, PINN is a highly flexible technique that allows incorporating all physical and even empirical knowledge about the system into the PINN structure. Additionally, the formulation of the optimization problem provides an opportunity to shape the estimated system, including information that can contribute to model generalization and the estimated system's physical correctness. Therefore, the optimization problem is formulated to parameterize the system\textcolor{blue}{~\eqref{systemtheta}}, aiming that this: i) mimics soil-water and \textcolor{blue}{evapotranspirative water depletion} dynamics;""",
    "PINN structure & opt",
)
text = tag_once(
    text,
    r"consistent with the strategy presented in \cite{condeforecast2024, Alessandri2017}.",
    r"consistent with the strategy presented in \textcolor{blue}{\cite{condeforecast2024, Alessandri2017}}.",
    "Data prep cite Alessandri",
)
text = tag_once(
    text,
    r"""To condition gradient-based optimization, all physical variables are normalized into \([0, 1]\) using fixed domain-scaling constants matched to operational field bounds: soil moisture by \SI{40}{\%VWC} (\(m_{\mathrm{scaled}} = m / 40\)), rainfall by \SI{30}{mm/h} (\(r_{\mathrm{scaled}} = r / 30\)), reference evapotranspiration by \SI{0.8}{mm/h} (\(ET_{0,\mathrm{scaled}} = ET_0 / 0.8\)), and irrigation duration fraction by \SI{1.0}{h/h} (\(q_{\mathrm{irr,scaled}} = q_{\mathrm{irr}} / 1\)), with hourly integration step \(\Delta t = \SI{1}{h}\). To clarify the estimation process, the following notation is introduced, where \( n_e \) is the size of the estimation window and \( t_k \) is the current time instant.

Let \( m(t_k) = [m_1(t_k), m_2(t_k), \dots, m_n(t_k)]^\top \in \mathbb{R}^{n} \) denote the measured soil moisture at $n$ depths ($n=6$) at the current time \( t_k \), and let \( m_i(t_{k_e \mid k}) \) denote the time series of measurements at depth \( i \) over an estimation window of size $n_e$, defined as  
${m_i(t_{k_e \mid k})}=\left[m_i(t_{k-n_e +1}), m_i(t_{k-n_e +2}), \dots, m_i(t_{k}) \right]^\top \in \mathbb{R}^{n_e}$;
then, the full set of measured soil moisture data over the estimation window is given by""",
    r"""\marginnote{R-2.9}\textcolor{blue}{To condition gradient-based optimization, all physical variables are normalized into \([0, 1]\) using fixed domain-scaling constants matched to operational field bounds: soil moisture by \SI{40}{\%VWC} (\(m_{\mathrm{scaled}} = m / 40\)), rainfall by \SI{30}{mm/h} (\(r_{\mathrm{scaled}} = r / 30\)), reference evapotranspiration by \SI{0.8}{mm/h} (\(ET_{0,\mathrm{scaled}} = ET_0 / 0.8\)), and irrigation duration fraction by \SI{1.0}{h/h} (\(q_{\mathrm{irr,scaled}} = q_{\mathrm{irr}} / 1\)), with hourly integration step \(\Delta t = \SI{1}{h}\).} To clarify the estimation process, the following notation is introduced, where \( n_e \) is the size of the estimation window and \( t_k \) is the current time instant.

Let \textcolor{blue}{\( m(t_k) = [m_1(t_k), m_2(t_k), \dots, m_n(t_k)]^\top \in \mathbb{R}^{n} \)} denote the measured soil moisture at $n$ depths \textcolor{blue}{($n=6$)} at the current time \( t_k \), and let \( m_i(t_{k_e \mid k}) \) \textcolor{blue}{denote} the time series of measurements at depth \( i \) over an estimation window of size $n_e$, defined as  
\textcolor{blue}{${m_i(t_{k_e \mid k})}=\left[m_i(t_{k-n_e +1}), m_i(t_{k-n_e +2}), \dots, m_i(t_{k}) \right]^\top \in \mathbb{R}^{n_e}$};
\textcolor{blue}{then}, the full set of measured soil moisture data over the estimation window is given by""",
    "Data prep 1",
)
text = tag_once(
    text,
    r"""Similarly, for running time index \( j \in \{k - n_e + 1, \dots, k\} \), we define the one-step finite-difference moisture rate \(\Delta m(t_j) \in \mathbb{R}^{n}\) as 
\[
\Delta m(t_j) = \frac{m(t_{j+1})-m(t_j)}{\Delta t},
\]
and \(\Delta \hat{m}(t_j) \in \mathbb{R}^{n}\) as 
\[
\Delta \hat{m}(t_j) = g(m(t_j),r(t_j),ET_0(t_j),q_{\mathrm{irr}}(t_j),\boldsymbol{\Theta}).
\]
The measured rain, irrigation, and computed potential evapotranspiration along the estimation process are defined as 
${r(t_{k_e \mid k})}=\left[r(t_{k-n_e +1}), r(t_{k-n_e +2}), \dots, r(t_{k}) \right]^\top \in \mathbb{R}^{n_e}$, 
${q_{\mathrm{irr}}(t_{k_e \mid k})}=\left[q_{\mathrm{irr}}(t_{k-n_e +1}), q_{\mathrm{irr}}(t_{k-n_e +2}), \dots, q_{\mathrm{irr}}(t_{k}) \right]^\top \in \mathbb{R}^{n_e}$,
and 
${ET_0(t_{k_e \mid k})}=\left[ET_0(t_{k-n_e +1}), ET_0(t_{k-n_e +2}), \dots, ET_0(t_{k}) \right]^\top \in \mathbb{R}^{n_e}$, 
respectively.""",
    r"""Similarly, \marginnote{R-1.9}{\color{blue}for running time index \( j \in \{k - n_e + 1, \dots, k\} \), we define the one-step finite-difference moisture rate \(\Delta m(t_j) \in \mathbb{R}^{n}\) as 
\[
\Delta m(t_j) = \frac{m(t_{j+1})-m(t_j)}{\Delta t},
\]
and \(\Delta \hat{m}(t_j) \in \mathbb{R}^{n}\) as 
\[
\Delta \hat{m}(t_j) = g(m(t_j),r(t_j),ET_0(t_j),q_{\mathrm{irr}}(t_j),\boldsymbol{\Theta}).
\]}
The measured rain, irrigation, and computed potential evapotranspiration along the estimation process are defined as 
\textcolor{blue}{${r(t_{k_e \mid k})}=\left[r(t_{k-n_e +1}), r(t_{k-n_e +2}), \dots, r(t_{k}) \right]^\top \in \mathbb{R}^{n_e}$}, 
\textcolor{blue}{${q_{\mathrm{irr}}(t_{k_e \mid k})}=\left[q_{\mathrm{irr}}(t_{k-n_e +1}), q_{\mathrm{irr}}(t_{k-n_e +2}), \dots, q_{\mathrm{irr}}(t_{k}) \right]^\top \in \mathbb{R}^{n_e}$},
and 
\textcolor{blue}{${ET_0(t_{k_e \mid k})}=\left[ET_0(t_{k-n_e +1}), ET_0(t_{k-n_e +2}), \dots, ET_0(t_{k}) \right]^\top \in \mathbb{R}^{n_e}$}, 
respectively.""",
    "Data prep 2",
)
text = tag_once(
    text,
    r"""\begin{equation}
    \dot{\hat{m}}(t) = g(\hat{m}, r, ET_0, q_{\mathrm{irr}}, \boldsymbol{\Theta}),
\end{equation}
with initial conditions \(\hat{m}(t_{k-n_e+1}) = m(t_{k-n_e+1})\), and solution over the window \(t \in [t_{k-n_e+1},\, t_k]\) given by
\begin{equation}
\label{integral}
    \hat{m}(t)
  \;=\;
  m(t_{k-n_e+1})\;+\;
  \int_{t_{k-n_e+1}}^{t}
        g\!\bigl(
            \hat{m}(\tau),
            r(\tau),
            ET_0(\tau),
            q_{\mathrm{irr}}(\tau),
            \boldsymbol{\Theta}
          \bigr)\, d\tau,
  \; t\in[t_{k-n_e+1},t_k].
\end{equation}
The state trajectory $ \hat{m}(t_{k_e \mid k}) $ is obtained at discrete time points $ t_{k - n_e + 1}, \dots, t_k $ (\(\Delta t = \SI{1}{h}\)). In practice, the integral in \eqref{integral} is approximated using numerical strategies such as direct multiple shooting, direct collocation, or successive linearizations. A simpler alternative, such as the Euler discretization, is less accurate and is therefore not considered here. We propose a direct multiple-shooting method that employs an explicit fourth-order Runge--Kutta (RK4) method with step \(\Delta t = \SI{1}{h}\) and a lower bound to prevent negative solutions for the estimated soil moisture.""",
    r"""\begin{equation}
    \dot{\hat{m}}(t) = g(\hat{m}, r, ET_0, \textcolor{blue}{q_{\mathrm{irr}}}, \boldsymbol{\Theta}),
\end{equation}
with initial conditions \(\hat{m}(t_{k-n_e+1}) = m(t_{k-n_e+1})\), and solution over the window \(t \in [t_{k-n_e+1},\, t_k]\) given by
\begin{equation}
\label{integral}
    \textcolor{blue}{\hat{m}(t)}
  \;=\;
  m(t_{k-n_e+1})\;+\;
  \int_{t_{k-n_e+1}}^{t}
        g\!\bigl(
            \hat{m}(\tau),
            r(\tau),
            \textcolor{blue}{ET_0(\tau)},
            \textcolor{blue}{q_{\mathrm{irr}}(\tau)},
            \boldsymbol{\Theta}
          \bigr)\, d\tau,
  \; t\in[t_{k-n_e+1},t_k].
\end{equation}
The state trajectory \textcolor{blue}{$ \hat{m}(t_{k_e \mid k}) $} is obtained at discrete time points $ t_{k - n_e + 1}, \dots, t_k $ \marginnote{R-2.9}\textcolor{blue}{(\(\Delta t = \SI{1}{h}\))}. In practice, the integral in \eqref{integral} is approximated using numerical strategies such as direct multiple shooting, direct collocation, or successive linearizations. A simpler alternative, such as the Euler discretization, is less accurate and is therefore not considered here. We propose a direct multiple-shooting method that employs an \textcolor{blue}{explicit fourth-order Runge--Kutta (RK4) method with step \(\Delta t = \SI{1}{h}\) and} a lower bound to prevent negative solutions for the estimated soil moisture.""",
    "Estimation integral RK4",
)
text = tag_once(
    text,
    r"Embedding closed-loop numerical integration directly inside optimization is well established in nonlinear system identification and moving horizon estimation (MHE) \cite{rawlings2017, Alessandri2017, Verhaegen2007, roffel2007process},",
    r"\marginnote{R-1.3d, R-1.9, R-2.1}{\color{blue}Embedding closed-loop numerical integration directly inside optimization is well established in nonlinear system identification and moving horizon estimation (MHE) \cite{rawlings2017, Alessandri2017, Verhaegen2007, roffel2007process},",
    "Two-stage start",
)
text = tag_once(
    text,
    "The $\\ell_1$ simulation norm balances large storm/irrigation pulses against slow evapotranspirative recession and suppresses multi-step drift \\cite{Aguirre2010}, as confirmed by Ablation~1 in Section~\\ref{sec:benchmarks}.\n\\end{enumerate}",
    "The $\\ell_1$ simulation norm balances large storm/irrigation pulses against slow evapotranspirative recession and suppresses multi-step drift \\cite{Aguirre2010}, as confirmed by Ablation~1 in Section~\\ref{sec:benchmarks}.\n\\end{enumerate}}",
    "Two-stage end",
)

# Penalizations (full tracking of all modified prose and equations)
text = tag_once(
    text,
    r"""\subsubsection{Penalizations}
To regularize the learned flux branches toward physically admissible behavior—non-negativity, monotonicity, and zero-forcing zero-flux—we augment both training stages with three soft penalty terms $c_1$, $c_2$, and $c_3$:
\[
\mathcal{L}_{\mathrm{total}}(\boldsymbol{\Theta}) = \lambda_1 j_1(\boldsymbol{\Theta}) + \lambda_2 j_2(\boldsymbol{\Theta}) + c_1(\boldsymbol{\Theta}) + c_2(\boldsymbol{\Theta}) + c_3(\boldsymbol{\Theta}),
\]
with $(\lambda_1, \lambda_2) = (1, 0)$ in Stage~1 and $(0, 1)$ in Stage~2. Crucially, because these conditions enter via finite penalty multipliers $(\lambda_{c_1}, \lambda_{c_2}, \lambda_{c_3})$ rather than hard architectural constraints, they act as \emph{soft physical regularizers} that strongly promote—while not algebraically forcing—exact positivity or monotonicity across every corner of the extrapolated input cube.

\textbf{Penalization Term \( c_1 \): Non-Negativity Regularizer}
To prevent unphysical negative drainage, infiltration, lateral outflow, or uptake, \( c_1 \) is evaluated over \( M = 1{,}000 \) uniform random collocation samples spanning the normalized domain \([0,1]^6 \times [0,1]^3\) at each optimization step:
\[
c_1 = \lambda_{c_1} \frac{1}{M} \sum_{k=1}^M \sum_{g \in \mathcal{G}} \left( \min(0, g(\mathbf{x}_k)) \right)^2,
\]
where \( \mathcal{G} = \{ f_1, \dots, f_{n-1}, fq_{\mathrm{irr}}, fr_1, \dots, fr_n, S_1, \dots, S_n, fl_1, \dots, fl_{n-1} \} \). Notably, both the lower boundary flux \( f_n \) and the inter-layer potential exchange terms \( f_{1,2}, \dots, f_{n-1,n} \) are intentionally excluded from \( c_1 \) so that \( f_{i,i+1}(m_i, m_{i+1}) \) remains signed (positive downward during infiltration/drainage and negative upward during capillary rise when \(m_{i+1} > m_i\)).

\textbf{Penalization Term \( c_2 \): Monotonicity Regularizer}
Evaluated over the same \( M = 1{,}000 \) uniform collocation points via automatic differentiation, \( c_2 \) encourages flux magnitudes to vary monotonically with their driving states and forcing variables:
\[
c_2 = \lambda_{c_2} \frac{1}{M} \sum_{k=1}^M \sum_{d \in \mathcal{D}} \left( \min(0, d(\mathbf{x}_k)) \right)^2,
\]
where
\[
\begin{aligned}
\mathcal{D} = \Biggl\{ &
\frac{\partial fq_{\mathrm{irr}}}{\partial q_{\mathrm{irr}}},\;
\frac{\partial fr_1}{\partial r}, \dots, \frac{\partial fr_n}{\partial r},\;
\frac{\partial f_1}{\partial m_1}, \dots, \frac{\partial f_n}{\partial m_n},\;
\frac{\partial f_{1,2}}{\partial (m_1 - m_2)}, \dots, \frac{\partial f_{n-1,n}}{\partial (m_{n-1} - m_n)}, \\
& \frac{\partial S_1}{\partial m_1}, \dots, \frac{\partial S_n}{\partial m_n},\;
\frac{\partial S_1}{\partial ET_0}, \dots, \frac{\partial S_n}{\partial ET_0},\;
\frac{\partial fl_1}{\partial m_1}, \dots, \frac{\partial fl_{n-1}}{\partial m_{n-1}}
\Biggr\}.
\end{aligned}
\]

\textbf{Penalization Term \( c_3 \): Zero-Forcing Zero-Flow Regularizer}
To ensure fluxes vanish when their driving forcing or moisture gradient is zero, \( c_3 \) evaluates squared flux residuals weighted by an indicator mask \( \delta_k \):
\[
c_3 = \lambda_{c_3} \frac{1}{M_b} \sum_{k=1}^{M_b} \sum_{g \in \mathcal{G}_0} g(\mathbf{x}_k)^2 \cdot \delta_k,
\]
where
\[
\mathcal{G}_0 = \bigl\{ f_1, \dots, f_{n-1},\; f_{1,2}, \dots, f_{n-1,n},\; fq_{\mathrm{irr}},\; fr_1, \dots, fr_n,\; S_1, \dots, S_n,\; fl_1, \dots, fl_{n-1} \bigr\},
\]
and \( \delta_k = 1 \) when the relevant driving input satisfies \( |\text{var}| \le \epsilon = 10^{-6} \) (e.g., \( |q_{\mathrm{irr}}| \le \epsilon \) for \( fq_{\mathrm{irr}} \), \( |r| \le \epsilon \) for \( fr_i \), \( |m_i - m_{i+1}| \le \epsilon \) for \( f_{i,i+1} \), or \( |m_i| \le \epsilon \) or \( |ET_0| \le \epsilon \) for \( S_i \)), and \( \delta_k = 0 \) otherwise. Importantly, in our implementation \( c_3 \) is evaluated directly on the hourly observed training batch (\( M_b = 839 \) steps) rather than uniform random samples over \( [0,1] \): because continuous random uniform sampling rarely hits within \( \epsilon = 10^{-6} \) of zero, evaluating \( c_3 \) on the operational training data naturally exploits the high frequency of exact zero forcing states in field records (non-rainy hours \( r=0 \), non-irrigated hours \( q_{\mathrm{irr}}=0 \), and nighttime \( ET_0=0 \)).

As detailed in the PINN structure above, single-input and inter-layer branches (\( f_i, f_{i,i+1}, fq_{\mathrm{irr}}, fr_i, fl_i \)) use depth \( L=4 \) (\( H_{\mathrm{b}}=H=56 \)), whereas two-input sink branches (\( S_i \)) use depth \( L=5 \) (\( H_{\mathrm{b}}=2H=112 \)). Penalty multipliers \( \lambda_{c_1}, \lambda_{c_2}, \lambda_{c_3} \) and base width \( H \) are selected via Bayesian hyperparameter optimization.""",
    r"""\subsubsection{Penalizations}
\marginnote{R-2.8}{\color{blue}To regularize the learned flux branches toward physically admissible behavior—non-negativity, monotonicity, and zero-forcing zero-flux—we augment both training stages with three soft penalty terms $c_1$, $c_2$, and $c_3$:
\[
\mathcal{L}_{\mathrm{total}}(\boldsymbol{\Theta}) = \lambda_1 j_1(\boldsymbol{\Theta}) + \lambda_2 j_2(\boldsymbol{\Theta}) + c_1(\boldsymbol{\Theta}) + c_2(\boldsymbol{\Theta}) + c_3(\boldsymbol{\Theta}),
\]
with $(\lambda_1, \lambda_2) = (1, 0)$ in Stage~1 and $(0, 1)$ in Stage~2. Crucially, because these conditions enter via finite penalty multipliers $(\lambda_{c_1}, \lambda_{c_2}, \lambda_{c_3})$ rather than hard architectural constraints, they act as \emph{soft physical regularizers} that strongly promote—while not algebraically forcing—exact positivity or monotonicity across every corner of the extrapolated input cube.}

\textbf{Penalization Term \( c_1 \): Non-Negativity \textcolor{blue}{Regularizer}}
\marginnote{R-2.8, R-2.9}\textcolor{blue}{To prevent unphysical negative drainage, infiltration, lateral outflow, or uptake, \( c_1 \) is evaluated over \( M = 1{,}000 \) uniform random collocation samples spanning the normalized domain \([0,1]^6 \times [0,1]^3\) at each optimization step:}
\[
c_1 = \lambda_{c_1} \frac{1}{M} \sum_{k=1}^M \sum_{g \in \mathcal{G}} \left( \min(0, g(\mathbf{x}_k)) \right)^2,
\]
where \textcolor{blue}{\( \mathcal{G} = \{ f_1, \dots, f_{n-1}, fq_{\mathrm{irr}}, fr_1, \dots, fr_n, S_1, \dots, S_n, fl_1, \dots, fl_{n-1} \} \). Notably, both the lower boundary flux \( f_n \) and the inter-layer potential exchange terms \( f_{1,2}, \dots, f_{n-1,n} \) are intentionally excluded from \( c_1 \) so that \( f_{i,i+1}(m_i, m_{i+1}) \) remains signed (positive downward during infiltration/drainage and negative upward during capillary rise when \(m_{i+1} > m_i\)).}

\textbf{Penalization Term \( c_2 \): Monotonicity \textcolor{blue}{Regularizer}}
\textcolor{blue}{Evaluated over the same \( M = 1{,}000 \) uniform collocation points via automatic differentiation, \( c_2 \) encourages flux magnitudes to vary monotonically with their driving states and forcing variables:}
\[
c_2 = \lambda_{c_2} \frac{1}{M} \sum_{k=1}^M \sum_{d \in \mathcal{D}} \left( \min(0, d(\mathbf{x}_k)) \right)^2,
\]
where
{\color{blue}\[
\begin{aligned}
\mathcal{D} = \Biggl\{ &
\frac{\partial fq_{\mathrm{irr}}}{\partial q_{\mathrm{irr}}},\;
\frac{\partial fr_1}{\partial r}, \dots, \frac{\partial fr_n}{\partial r},\;
\frac{\partial f_1}{\partial m_1}, \dots, \frac{\partial f_n}{\partial m_n},\;
\frac{\partial f_{1,2}}{\partial (m_1 - m_2)}, \dots, \frac{\partial f_{n-1,n}}{\partial (m_{n-1} - m_n)}, \\
& \frac{\partial S_1}{\partial m_1}, \dots, \frac{\partial S_n}{\partial m_n},\;
\frac{\partial S_1}{\partial ET_0}, \dots, \frac{\partial S_n}{\partial ET_0},\;
\frac{\partial fl_1}{\partial m_1}, \dots, \frac{\partial fl_{n-1}}{\partial m_{n-1}}
\Biggr\}.
\end{aligned}
\]}

\textbf{Penalization Term \( c_3 \): Zero-\textcolor{blue}{Forcing Zero-Flow Regularizer}}
\marginnote{R-2.7}{\color{blue}To ensure fluxes vanish when their driving forcing or moisture gradient is zero, \( c_3 \) evaluates squared flux residuals weighted by an indicator mask \( \delta_k \):
\[
c_3 = \lambda_{c_3} \frac{1}{M_b} \sum_{k=1}^{M_b} \sum_{g \in \mathcal{G}_0} g(\mathbf{x}_k)^2 \cdot \delta_k,
\]
where
\[
\mathcal{G}_0 = \bigl\{ f_1, \dots, f_{n-1},\; f_{1,2}, \dots, f_{n-1,n},\; fq_{\mathrm{irr}},\; fr_1, \dots, fr_n,\; S_1, \dots, S_n,\; fl_1, \dots, fl_{n-1} \bigr\},
\]
and \( \delta_k = 1 \) when the relevant driving input satisfies \( |\text{var}| \le \epsilon = 10^{-6} \) (e.g., \( |q_{\mathrm{irr}}| \le \epsilon \) for \( fq_{\mathrm{irr}} \), \( |r| \le \epsilon \) for \( fr_i \), \( |m_i - m_{i+1}| \le \epsilon \) for \( f_{i,i+1} \), or \( |m_i| \le \epsilon \) or \( |ET_0| \le \epsilon \) for \( S_i \)), and \( \delta_k = 0 \) otherwise. Importantly, in our implementation \( c_3 \) is evaluated directly on the hourly observed training batch (\( M_b = 839 \) steps) rather than uniform random samples over \( [0,1] \): because continuous random uniform sampling rarely hits within \( \epsilon = 10^{-6} \) of zero, evaluating \( c_3 \) on the operational training data naturally exploits the high frequency of exact zero forcing states in field records (non-rainy hours \( r=0 \), non-irrigated hours \( q_{\mathrm{irr}}=0 \), and nighttime \( ET_0=0 \)).}

\marginnote{R-2.9}\textcolor{blue}{As detailed in the PINN structure above, single-input and inter-layer branches (\( f_i, f_{i,i+1}, fq_{\mathrm{irr}}, fr_i, fl_i \)) use depth \( L=4 \) (\( H_{\mathrm{b}}=H=56 \)), whereas two-input sink branches (\( S_i \)) use depth \( L=5 \) (\( H_{\mathrm{b}}=2H=112 \)). Penalty multipliers \( \lambda_{c_1}, \lambda_{c_2}, \lambda_{c_3} \) and base width \( H \) are selected via Bayesian hyperparameter optimization.}""",
    "Penalizations full",
)

# 5. Case Study, Computational Tools, Code & Data Availability
text = tag_once(
    text,
    r"The site's Classical Alfisols are poorly drained, featuring a shallow sandy topsoil over a low-permeability argillic clay horizon at approximately \SI{30}{cm} depth that restricts deep percolation and confines feeder root development primarily to the upper \SI{30}{cm}, a condition exacerbated by seawater intrusion. As illustrated in Fig.~\ref{fig:bed_ditch}, trees are planted on raised beds flanked by open drainage ditches that lower local water tables, mitigate salinity from occasional brackish intrusion, and create lateral subsurface drainage pathways directly below the shallow root zone—explaining both direct multi-depth rainfall infiltration along sloped bed shoulders and lateral ditch-level moisture exchange observed at deeper probe sensors. Irrigation uses under-tree micro-sprinklers",
    r"The site's Classical Alfisols are poorly drained, \marginnote{R-1.4}\textcolor{blue}{featuring a shallow sandy topsoil over a low-permeability argillic clay horizon at approximately \SI{30}{cm} depth that restricts deep percolation and confines feeder root development primarily to the upper \SI{30}{cm}}, a condition exacerbated by seawater intrusion. \textcolor{blue}{As illustrated in Fig.~\ref{fig:bed_ditch}, trees are planted on raised beds flanked by open drainage ditches that lower local water tables, mitigate salinity from occasional brackish intrusion, and create lateral subsurface drainage pathways directly below the shallow root zone—explaining both direct multi-depth rainfall infiltration along sloped bed shoulders and lateral ditch-level moisture exchange observed at deeper probe sensors.} Irrigation uses \textcolor{blue}{under-tree micro-}sprinklers",
    "Case study bed ditch",
)
text = tag_once(
    text,
    r"\caption{Cross-sectional geometry (not to scale) of the IRREC Millennium Block experimental site (35-acre block; 20-acre active 5,500-tree citrus trial), showing a citrus tree planted on a raised sandy Alfisol bed flanked by drainage ditches above an argillic horizon at $\approx\SI{30}{cm}$ depth. A 6-depth Sentek Drill \& Drop capacitive probe (\SI{20}{cm} from the trunk) monitors 10-\si{cm} vertical compartments centered at 5, 15, 25, 35, 45, and 55~\si{cm}, alongside micro-sprinkler irrigation ($q_{\mathrm{irr}}$), rainfall ($r$), evapotranspiration ($ET_0$), vertical percolation ($f_i$), inter-layer exchange ($f_{i,i+1}$), and lateral ditch drainage ($fl_1\dots fl_5$).}",
    r"\marginnote{R-1.4}\caption{\textcolor{blue}{Cross-sectional geometry (not to scale) of the IRREC Millennium Block experimental site (35-acre block; 20-acre active 5,500-tree citrus trial), showing a citrus tree planted on a raised sandy Alfisol bed flanked by drainage ditches above an argillic horizon at $\approx\SI{30}{cm}$ depth. A 6-depth Sentek Drill \& Drop capacitive probe (\SI{20}{cm} from the trunk) monitors 10-\si{cm} vertical compartments centered at 5, 15, 25, 35, 45, and 55~\si{cm}, alongside micro-sprinkler irrigation ($q_{\mathrm{irr}}$), rainfall ($r$), evapotranspiration ($ET_0$), vertical percolation ($f_i$), inter-layer exchange ($f_{i,i+1}$), and lateral ditch drainage ($fl_1\dots fl_5$).}}",
    "Fig 6 caption",
)
text = tag_once(
    text,
    r"\caption{Description of experimental setup, where a weather station, soil moisture probe, and irrigation sensor were deployed to collect soil moisture ($m$), rain ($r$), potential evapotranspiration ($ET_0$), and irrigation ($q_{\mathrm{irr}}$) data.}",
    r"\marginnote{R-1.10}\caption{Description of experimental setup, where a weather station, soil moisture probe, and irrigation sensor were deployed to collect soil moisture ($m$), rain ($r$), potential evapotranspiration ($ET_0$), and irrigation (\textcolor{blue}{$q_{\mathrm{irr}}$}) data.}",
    "Fig 7 q_irr",
)
text = tag_once(
    text,
    r"""\item[Weather Station:] A Davis Vantage Pro2 weather station\footnote{See \url{https://www.davisinstruments.com/pages/vantage-pro2} for more information on Davis weather stations.} providing rainfall (\si{mm/h}) and meteorological variables (temperature, humidity, wind speed, solar radiation, and barometric pressure). These data were employed to calculate $ET_0$ (\si{mm/h}) using the FAO-56 Penman--Monteith method \cite{allen1998crop}.
\item[Soil Moisture Probe:] SM measurements ($m_1\dots m_6$ in \si{\%VWC}) were taken at 5\si{cm}, 15\si{cm}, 25\si{cm}, 35\si{cm}, 45\si{cm}, 55\si{cm} using a Sentek Drill \& Drop probe\footnote{Series II Sentek Drill \& Drop probe, capable of measuring SM, temperature, and salinity at 10\si{cm} intervals. See \url{https://sentektechnologies.com/download/sentek-drill-drop-probe-manual/} for details.}, located 20\si{cm} from the base of the tree.
\item[Irrigation Sensor:] Irrigation events were detected by installing two electrodes at the supply irrigation point, connected to a Campbell Scientific CR1000X datalogger, where changes in electrical conduction were used to identify irrigation events and their hourly duration fraction ($q_{\mathrm{irr}} \in [0, 1]\,\si{h/h}$).""",
    r"""\item[Weather Station:] A Davis Vantage Pro2 weather station\footnote{See \url{https://www.davisinstruments.com/pages/vantage-pro2} for more information on Davis weather stations.} providing rainfall \marginnote{R-1.2}\textcolor{blue}{(\si{mm/h})} and meteorological variables (temperature, humidity, wind speed, solar radiation, and barometric pressure). These data were employed to calculate $ET_0$ \textcolor{blue}{(\si{mm/h})} using the \textcolor{blue}{FAO-56 Penman--Monteith} method \cite{allen1998crop}.
\item[Soil Moisture Probe:] SM measurements \textcolor{blue}{($m_1\dots m_6$ in \si{\%VWC})} were taken at 5\si{cm}, 15\si{cm}, 25\si{cm}, 35\si{cm}, 45\si{cm}, 55\si{cm} using a Sentek Drill \& Drop probe\footnote{Series II Sentek Drill \& Drop probe, capable of measuring SM, temperature, and salinity at 10\si{cm} intervals. See \url{https://sentektechnologies.com/download/sentek-drill-drop-probe-manual/} for details.}, located 20\si{cm} from the base of the tree.
\item[Irrigation Sensor:] Irrigation events were detected by installing two electrodes at the supply irrigation point, connected to a Campbell Scientific CR1000X datalogger, where changes in electrical conduction were used to identify irrigation events and their \textcolor{blue}{hourly duration fraction ($q_{\mathrm{irr}} \in [0, 1]\,\si{h/h}$)}.""",
    "Sensors list",
)
text = tag_once(
    text,
    r"The training dataset comprises 840 hourly timestamps spanning 839 elapsed hours ($\approx 35$~days, producing 839 one-hour finite-difference transitions) from March 28, 2025, 21:00:00 to May 2, 2025, 20:00:00, encompassing roughly seven distinct rainfall events and multiple scheduled irrigation cycles (Fig. \ref{Mainfig}).",
    r"\marginnote{R-2.4}\textcolor{blue}{The training dataset comprises 840 hourly timestamps spanning 839 elapsed hours ($\approx 35$~days, producing 839 one-hour finite-difference transitions) from March 28, 2025, 21:00:00 to May 2, 2025, 20:00:00, encompassing roughly seven distinct rainfall events and multiple scheduled irrigation cycles} (Fig. \ref{Mainfig}).",
    "Training dates",
)
text = tag_once(
    text,
    r"Raw validation sensor logs span 1,008 hourly records from April 24 to June 5, 2025. To ensure strict temporal separation after training, we extract the held-out 3-day validation benchmark via \(t_{\mathrm{val}} \in [216, 288]\,\si{h}\) (73 hourly records spanning 72 elapsed hours, from May 3, 2025, 21:00:00 to May 6, 2025, 21:00:00), leaving a 25-hour separation buffer before the initial validation state $m(0)$ on May 3 at 21:00:00; the full 792 unseen hours through June 5, 2025, 20:00:00 are evaluated in Section~\ref{sec:benchmarks} across 30 rolling 3-day windows and a 7-day rollout. During all held-out forecasting evaluations, observed station rainfall ($r$), $ET_0$, and irrigation ($q_{\mathrm{irr}}$) series are supplied as known external inputs to isolate intrinsic model dynamics free of numerical weather prediction uncertainty; in live scheduling deployment, operational National Weather Service forecast feeds \cite{NWS2024} replace historical weather inputs.",
    r"\marginnote{R-2.3, R-2.4}\textcolor{blue}{Raw validation sensor logs span 1,008 hourly records from April 24 to June 5, 2025. To ensure strict temporal separation after training, we extract the held-out 3-day validation benchmark via \(t_{\mathrm{val}} \in [216, 288]\,\si{h}\) (73 hourly records spanning 72 elapsed hours, from May 3, 2025, 21:00:00 to May 6, 2025, 21:00:00), leaving a 25-hour separation buffer before the initial validation state $m(0)$ on May 3 at 21:00:00; the full 792 unseen hours through June 5, 2025, 20:00:00 are evaluated in Section~\ref{sec:benchmarks} across 30 rolling 3-day windows and a 7-day rollout. During all held-out forecasting evaluations, observed station rainfall ($r$), $ET_0$, and irrigation ($q_{\mathrm{irr}}$) series are supplied as known external inputs to isolate intrinsic model dynamics free of numerical weather prediction uncertainty; in live scheduling deployment, operational National Weather Service forecast feeds \cite{NWS2024} replace historical weather inputs.}",
    "Validation dates & observed weather",
)
text = tag_once(
    text,
    r"The numerical integration of the PINN’s differential equations was performed using a custom RK4 solver (\(\Delta t = \SI{1}{h}\)) fully integrated into TensorFlow’s computational graph. Randomness across initialization and uniform collocation sampling (\(M = 1{,}000\) points per step) was controlled using fixed seed \texttt{42} across Python, NumPy, and TensorFlow. Both stages used AMSGrad-Adam optimization with gradient norm/value clipping at 0.5 and staircase exponential decay (factor 0.9 every 1,000 steps): Stage~1 ran for 3,000 epochs starting from \(\eta_1 = \num{7.97e-5}\), and Stage~2 refined parameters via RK4 trajectory tracking for 1,000 epochs starting from \(\eta_2 = \num{5.0e-6}\).",
    r"The numerical integration of the PINN’s differential equations was performed using a custom RK4 solver \marginnote{R-2.9}\textcolor{blue}{(\(\Delta t = \SI{1}{h}\))} fully integrated into TensorFlow’s computational graph\textcolor{blue}{. Randomness across initialization and uniform collocation sampling (\(M = 1{,}000\) points per step) was controlled using fixed seed \texttt{42} across Python, NumPy, and TensorFlow. Both stages used AMSGrad-Adam optimization with gradient norm/value clipping at 0.5 and staircase exponential decay (factor 0.9 every 1,000 steps): Stage~1 ran for 3,000 epochs starting from \(\eta_1 = \num{7.97e-5}\), and Stage~2 refined parameters via RK4 trajectory tracking for 1,000 epochs starting from \(\eta_2 = \num{5.0e-6}\)}.",
    "Setup seed Stage 2",
)
text = tag_once(
    text,
    r"""\item Initial learning rate for Stage 1 ($\eta_1$): \num{7.97e-5} (and Stage 2 $\eta_2 = \num{5.0e-6}$).
\item Number of neurons per layer ($H$): 56 (with $2H = 112$ on two-input sink branches $S_i$).""",
    r"""\item Initial learning rate for Stage 1 ($\eta_1$): \num{7.97e-5} \textcolor{blue}{(and Stage 2 $\eta_2 = \num{5.0e-6}$)}.
\item Number of neurons per layer ($H$): 56 \textcolor{blue}{(with $2H = 112$ on two-input sink branches $S_i$)}.""",
    "Setup bullets",
)
text = tag_once(
    text,
    "\\section*{Code availability}\nAll custom Python/TensorFlow code developed for this study—including the two-stage PINN training pipeline, branched flux-network definitions, custom \\texttt{float64} RK4 solver, inference scripts, trained model weights, statistical verification utilities, and the complete benchmark, ablation, and physical admissibility evaluation scripts—is openly deposited in Zenodo at \\url{https://doi.org/10.5281/zenodo.19446129}.\n\n\\section*{Data availability}\nThe complete raw and hourly processed field sensor datasets (weather station, multi-depth soil moisture probe, and irrigation conductance telemetry) supporting all training, 3-day benchmark, and rolling unseen-window analyses are publicly deposited in Zenodo at \\url{https://doi.org/10.5281/zenodo.19446129}.\n\n\\section*{Author contributions}\nG.C. conceived the methodology, developed the PINN software and numerical verification experiments, analyzed the field data, and drafted the manuscript. S.M.G. supervised the research, acquired funding, designed the irrigation monitoring infrastructure, and contributed to manuscript revision. J.I.P. contributed to the control-theoretic, system-identification, and optimization formulation and manuscript review. F.T.Z. managed the Millennium Block citrus experimental site, provided agronomic interpretation, and reviewed the manuscript. All authors read and approved the final manuscript.\n\n\\section*{Competing interests}\nThe authors declare no competing interests.",
    "{\\color{blue}\\section*{Code availability}\\marginnote{E-1.2}\nAll custom Python/TensorFlow code developed for this study—including the two-stage PINN training pipeline, branched flux-network definitions, custom \\texttt{float64} RK4 solver, inference scripts, trained model weights, statistical verification utilities, and the complete benchmark, ablation, and physical admissibility evaluation scripts—is openly deposited in Zenodo at \\url{https://doi.org/10.5281/zenodo.19446129}.\n\n\\section*{Data availability}\nThe complete raw and hourly processed field sensor datasets (weather station, multi-depth soil moisture probe, and irrigation conductance telemetry) supporting all training, 3-day benchmark, and rolling unseen-window analyses are publicly deposited in Zenodo at \\url{https://doi.org/10.5281/zenodo.19446129}.\n\n\\section*{Author contributions}\nG.C. conceived the methodology, developed the PINN software and numerical verification experiments, analyzed the field data, and drafted the manuscript. S.M.G. supervised the research, acquired funding, designed the irrigation monitoring infrastructure, and contributed to manuscript revision. J.I.P. contributed to the control-theoretic, system-identification, and optimization formulation and manuscript review. F.T.Z. managed the Millennium Block citrus experimental site, provided agronomic interpretation, and reviewed the manuscript. All authors read and approved the final manuscript.\n\n\\section*{Competing interests}\nThe authors declare no competing interests.}",
    "Code/Data avail",
)

with open(COMM_PATH, "w", encoding="utf-8") as f:
    f.write(text)

# Automated integrity verification against 03_24_2026_paper_original.tex and PINN_revised_paper_clean.tex
with open(ORIG_PATH, "r", encoding="utf-8") as f:
    orig_src = f.read()
with open(CLEAN_PATH, "r", encoding="utf-8") as f:
    clean_src = f.read()


def verify_tracking(comm_s, clean_s, orig_s):
    # 1. Exact round-trip identity with clean_s
    st = comm_s.replace(preamble_repl, preamble_target, 1)
    while r"\marginnote{" in st:
        i = st.index(r"\marginnote{")
        depth, j = 1, i + len(r"\marginnote{")
        while j < len(st) and depth > 0:
            if st[j] == "{" and st[j - 1] != "\\":
                depth += 1
            elif st[j] == "}" and st[j - 1] != "\\":
                depth -= 1
            j += 1
        st = st[:i] + st[j:]
    for cmd in (r"\textcolor{blue}{", r"{\color{blue}"):
        while cmd in st:
            i = st.index(cmd)
            depth, j = 1, i + len(cmd)
            while j < len(st) and depth > 0:
                if st[j] == "{" and st[j - 1] != "\\":
                    depth += 1
                elif st[j] == "}" and st[j - 1] != "\\":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            st = st[:i] + st[i + len(cmd) : j] + st[j + 1 :]
    assert st == clean_s, "Stripped commented text does not match clean text!"

    # 2. Check all non-structural tokens in body against orig_s
    n = len(comm_s)
    is_blue = [False] * n
    is_markup = [False] * n
    i = 0
    while i < n:
        if comm_s.startswith(r"\marginnote{", i):
            for k in range(i, i + len(r"\marginnote{")):
                is_markup[k] = True
            depth, j = 1, i + len(r"\marginnote{")
            while j < n and depth > 0:
                is_markup[j] = True
                if comm_s[j] == "{" and comm_s[j - 1] != "\\":
                    depth += 1
                elif comm_s[j] == "}" and comm_s[j - 1] != "\\":
                    depth -= 1
                j += 1
            i = j
        else:
            i += 1
    for cmd in (r"\textcolor{blue}{", r"{\color{blue}"):
        i = 0
        while i < n:
            if comm_s.startswith(cmd, i):
                for k in range(i, i + len(cmd)):
                    is_markup[k] = True
                depth, j = 1, i + len(cmd)
                while j < n and depth > 0:
                    if comm_s[j] == "{" and comm_s[j - 1] != "\\":
                        depth += 1
                    elif comm_s[j] == "}" and comm_s[j - 1] != "\\":
                        depth -= 1
                        if depth == 0:
                            is_markup[j] = True
                            break
                    if not is_markup[j]:
                        is_blue[j] = True
                    j += 1
                i += len(cmd)
            else:
                i += 1
    stripped, meta = [], []
    ln = 1
    for idx, ch in enumerate(comm_s):
        if ch == "\n":
            ln += 1
        if not is_markup[idx]:
            stripped.append(ch)
            meta.append((ln, is_blue[idx]))
    s_body = "".join(stripped)
    word_re = re.compile(r"\\[a-zA-Z@]+\*?|[A-Za-z0-9]+|\S")
    o_toks = [m.group(0) for m in word_re.finditer(orig_s, orig_s.index(r"\abstract{"))]
    c_toks = [
        (m.group(0), meta[m.start()][0], any(meta[k][1] for k in range(m.start(), m.end())))
        for m in word_re.finditer(s_body, s_body.index(r"\abstract{"))
    ]
    sm = difflib.SequenceMatcher(None, o_toks, [t[0] for t in c_toks], autojunk=False)
    structural_ok = {
        r"\begin", r"\end", "table", "figure", "tablenotes", r"\centering",
        r"\caption", r"\label", r"\scriptsize", r"\tiny", r"\footnotesize",
        r"\fontsize", r"\selectfont",
        r"\setlength", r"\tabcolsep", r"\item", r"\includegraphics",
        r"\linewidth", r"\FloatBarrier", "{", "}", "[", "]", "!", "htbp", "h",
        "2pt", "1", ".", "8pt", "2", "5pt", "6", "7", "8", "0", "96", "92", ":", "_", ";", "\\", "tab", "lit",
        "comparison", "fig", "scatter", "parity", "bed", "ditch", "schematic",
        "predicted", "vs", "measured", "pdf", "=", "Results", "width",
    }
    bad = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for tok, ln, blue in c_toks[j1:j2]:
            if not blue and tok not in structural_ok:
                bad.append((ln, tok))
    assert not bad, f"Untracked modified/inserted prose tokens found: {bad[:20]}"


verify_tracking(text, clean_src, orig_src)
print("Successfully wrote & verified 100% tracked:", COMM_PATH, "length:", len(text))

