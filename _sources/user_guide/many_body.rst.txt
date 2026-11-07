.. _many-body:

*****************
Many-body effects
*****************

Each EDA number is an *interaction* energy of a set of fragments, so a fragment alone
contributes nothing. From the whole cluster and its pairs the step gives, for every term:

2-body
    The sum of the term over all pairs of fragments, each pair computed alone.

≥3-body (exact)
    The whole cluster's term minus the 2-body sum. It contains everything beyond pairs:
    3-body, 4-body and higher. It is exact, but needs the cluster and every pair.

If triples are computed as well:

3-body (triples)
    The 3-body increment of a triple *ijk* is

    .. math::

       \Delta E_{ijk} = E_{ijk} - E_{ij} - E_{ik} - E_{jk}

    and the step sums it over the triples that were computed -- all of them, or those
    within the cutoff. **This is a truncated sum, not the ≥3-body total.** Even with all
    triples it leaves out the 4-body and higher parts, which are the difference between
    the two.

The two are always reported under different names (``≥3-body (exact)`` and
``3-body (triples)``) so that one cannot be read as the other.

What to expect
==============
For a Li⁺ ion with its first solvation shell (Li⁺, 2 EC, 2 DMC; ωB97M-V/def2-TZVPD) the
≥3-body part of the total is +171 kJ/mol, against a 2-body sum of -667 kJ/mol. Almost
all of it is polarization (+160 kJ/mol): the solvent molecules polarize each other less
in the cluster than the Li⁺ polarizes each of them alone. Classical electrostatics has
no ≥3-body part, as it must, so the frozen term's -1.1 kJ/mol is +4.6 kJ/mol of
dispersion and -5.7 kJ/mol of (common) Pauli.
