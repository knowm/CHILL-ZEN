# CHILL ZEN — the run order, as targets.
#
# The grayscale main sequence, experiments 00-14:
#
#   make artifacts   fit the codecs and teach the banks (the long part)
#   make verdict     the two-level verdict behind the paper's Table II
#   make analysis    the prefix schedule and the draw's signature
#   make energy      the energy model and the frontier
#   make figures     every figure that does not need the head-to-head
#   make all         those five, in order
#
# Everything else is its own command, because each costs hours or needs
# artifacts the main sequence does not build:
#
#   make sweep       the backbone codes (books x symbols @ keep-p) the scaling study compares (hours)
#   make binary      the binary-trained arms (17-19, 22)
#   make comparator  the comparator-noise sweeps               (after binary)
#   make comparator-figure       Fig. 4 (after the sweeps are scored)
#   make physical    the physical-operating-point rows        (after binary)
#   make audit       the free-label draw of Sec. V C          (after binary)
#   make cold-control  the cold sweeps at the physical floor (experiment 35)
#   make dose-chain    the joint sweep chained to eight doses (experiment 37)
#   make ablations   the draw ablations of experiment 20
#   make reporting   Table II's seed spread and the flip rate
#   make head-to-head-figures    Figs. 7 and 8 (after the head-to-head)
#
# "after X" is a note, not a prerequisite: these targets do not trigger
# the hours of work they depend on, so run them in the order shown.
#
# The head-to-head has its own environment and its own sequence; see
# experiments/head_to_head/README.md.
#
#   make paper       build main.pdf from main.tex
#
# The paper build needs tectonic (brew install tectonic), chosen because it
# fetches revtex4-2 and runs the bib pass itself.

PY ?= .venv/bin/python
E  = experiments
TECTONIC ?= tectonic

.PHONY: all paper artifacts sweep verdict analysis energy figures \
        ablations reporting binary comparator comparator-figure physical \
        audit cold-control dose-chain head-to-head-figures clean-outputs \
        clean-paper

all: artifacts verdict analysis energy figures

paper: main.pdf

main.pdf: main.tex refs.bib
	$(TECTONIC) main.tex

artifacts:
	$(PY) $(E)/00_train_critic.py
	$(PY) $(E)/01_fit_backbone_codec.py
	$(PY) $(E)/02_fit_patch_codec.py
	$(PY) $(E)/03_teach_backbone_banks.py
	$(PY) $(E)/04_teach_patch_banks.py
	$(PY) $(E)/05_teach_joint_bank.py

sweep:
	$(PY) $(E)/01_fit_backbone_codec.py --all
	$(PY) $(E)/03_teach_backbone_banks.py --all
	$(PY) $(E)/06_backbone_sweep.py --all

verdict:
	$(PY) $(E)/06_backbone_sweep.py
	$(PY) $(E)/07_verdict.py
	$(PY) $(E)/08_stack_arms.py

analysis:
	$(PY) $(E)/09_prefix_schedule.py
	$(PY) $(E)/10_draw_analysis.py

energy:
	$(PY) $(E)/11_operating_point.py
	$(PY) $(E)/12_energy_model.py
	$(PY) $(E)/13_energy_frontier.py
	$(PY) $(E)/36_sneak_paths.py
	$(PY) $(E)/38_readout_options.py

figures:
	$(PY) $(E)/14_figures.py

ablations:
	$(PY) $(E)/20_draw_ablations.py

reporting:
	$(PY) $(E)/21_reporting.py

# The binary-trained arms; scoring runs in the head-to-head's own
# environment (experiments/head_to_head/06_binary_arm.py).
binary:
	$(PY) $(E)/17_fit_binary_codec.py
	$(PY) $(E)/18_teach_binary_banks.py
	$(PY) $(E)/19_binary_verdict.py
	$(PY) $(E)/22_binary_energy.py

# The comparator-noise sweeps. 24 needs the binary banks (make binary),
# and runs twice: the declared grid, then --extend for P6-P7. Its FID
# scoring is a separate step in the head-to-head's environment:
#
#   .venv-dtm/bin/python experiments/head_to_head/07_comparator_score.py
#
# Fig. 4 is a separate target because it reads that scoring pass. Run it
# after the line above, or it draws the previous scoring's binary curve.
comparator:
	$(PY) $(E)/23_comparator_sweep.py
	$(PY) $(E)/24_binary_comparator_sweep.py
	$(PY) $(E)/24_binary_comparator_sweep.py --extend

comparator-figure:
	$(PY) $(E)/29_comparator_figure.py

# Table VII's physical-point rows and the energy restatement. 25, 34 and
# 27 render for scoring in the head-to-head's environment
# (07_comparator_score.py, which prints each grayscale arm's best level);
# 27 and the binary rows need the binary banks.
physical:
	$(PY) $(E)/head_to_head/25_comparator_headtohead.py
	$(PY) $(E)/head_to_head/34_grayscale_grid.py
	$(PY) $(E)/27_binary_comparator_headtohead.py
	$(PY) $(E)/28_physical_energy.py

# The free-label draw of the Sec. V C audit; scored by
# head_to_head/32_audit_score.py in the head-to-head's environment.
audit:
	$(PY) $(E)/31_free_draw.py

# The cold-read control of Sec. IV C: both cold sweeps through the
# physical read at 50 mV and the register's floor, against the noiseless
# argmax the deployed generation uses. Needs only the deployed banks.
cold-control:
	$(PY) $(E)/35_cold_noise_control.py

# The joint sweep as a chain (Sec. III B): eight doses under six sweep
# read conditions, then noisy chains quenched by one cold dose.
dose-chain:
	$(PY) $(E)/37_dose_chain.py
	$(PY) $(E)/37_dose_chain.py --quench

# These two need experiments/head_to_head to have run.
head-to-head-figures:
	$(PY) $(E)/15_energy_figure.py
	$(PY) $(E)/16_binarization_figure.py

clean-outputs:
	rm -rf logs

clean-paper:
	rm -f main.pdf
