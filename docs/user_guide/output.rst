.. _output:

******
Output
******

Printed output
==============
The step prints the fragments with their charges, a table of every native and common
term for the whole cluster with its 2-body, ≥3-body (exact) and, if computed, 3-body
(triples) parts, a table of the pairs' common terms, and the conventions the numbers
follow.

Results
=======
Results that can be put in variables or tables, all in kJ/mol:

* ``EDA <term>``: the whole cluster's term;
* ``EDA <term> 2-body``, ``EDA <term> ≥3-body`` and ``EDA <term> 3-body (triples)``;
* ``EDA pairs`` and ``EDA triples``: each pair's terms and each triple's 3-body
  increments;
* ``EDA method``: the scheme, code and settings.

``<term>`` is one of the common terms (see :ref:`eda-terms`): electrostatics, Pauli,
dispersion, frozen, polarization, charge transfer, induction, total. Electrostatics is
the classical electrostatics and Pauli is frozen − classical electrostatics −
dispersion, the pair comparable with SAPT. The cluster's terms
and their ≥3-body parts are also stored as properties of the configuration, named for the
method, e.g. ``EDA total#Interaction EDA#ALMO-EDA2/wB97M-V/def2-TZVPD``.

eda.extxyz
==========
One frame for each set of fragments computed (the cluster, then the pairs, then any
triples), with the atoms' fragment and fragment charge as columns, and these keys:

* ``EDA_ALMOEDA2_<term>``: the set's own terms in the scheme's names
  (``electrostatic``, ``classical_electrostatic``, ``pauli``, ``dispersion``,
  ``frozen``, ``polarization``, ``charge_transfer``, ``total``), ``induction`` and
  ``mod_pauli``. ``electrostatic`` and ``pauli`` are the orthogonal-decomposition
  pair, which is not comparable with SAPT; the SAPT-comparable pair is
  ``classical_electrostatic`` and ``mod_pauli`` = frozen − classical_electrostatic −
  dispersion (Q-Chem's CLS ELEC and MOD PAULI);
* on the cluster's frame, ``..._<term>_2body``, ``..._<term>_ge3body_exact`` and, with
  triples, ``..._<term>_3body_triples`` (truncated);
* on each triple's frame, ``..._<term>_3body``, its 3-body increment;
* ``EDA_method``: e.g. ``scheme=ALMO-EDA2;backend=GPU4PySCF;xc=wb97m-v;
  basis=def2-tzvpd;auxbasis=none;nlc=default;response_nlc=on;disp=none;ferf=coupled;
  svd_threshold=0.0001``;
* ``EDA_conventions``: the units, the geometry convention (no distortion term), that
  charge transfer carries the BSSE, what induction is, which electrostatic and Pauli
  terms are comparable with SAPT, and which many-body sum is truncated.

Every frame also has ``system_name`` and ``configuration_name``, the structure it came
from. With several structures, eda.extxyz has the frames of all of them.

eda.json and eda.jsonl
======================
Everything about a structure: its system and configuration names (``structure``), the
method and settings, the fragments (name, charge, and the 0-based indices of their
atoms), each calculation's terms with its wall time, peak GPU memory, GPU and code
versions (or its error), and the many-body sums. One structure gives ``eda.json``;
several give ``eda.jsonl``, one line per structure in the same format.

Tasks and reruns
================
Each calculation runs in its own directory, ``tasks/<key>``, where the key is the set
of fragments (``0+1``) or, with several structures, the structure's number and the set
(``3.0+1``); its input, the code's log and its ``results.jsonl`` are there. Finished
calculations are marked, so if a job stops -- for example it runs out of time --
rerunning it computes only the others. A finished calculation is kept as long as
what determines its results is unchanged: the code, the method settings and the
fragments' atoms. The memory, the threads, and newer versions of the step that do
not change the results do not make it run again. A failed calculation is reported,
tried again on a rerun, and the step still gives every part that does not need it.
