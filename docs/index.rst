Welcome to the documentation for the Interaction EDA SEAMM plug-in
==================================================================

The Interaction EDA step splits the interaction energy between the molecules of a
cluster into physically meaningful terms -- electrostatics, Pauli repulsion,
dispersion, polarization and charge transfer -- with an energy decomposition analysis
(EDA). It computes the whole cluster and its pairs (and optionally triples), and
separates each term into its 2-body and many-body parts. The first release uses
second-generation ALMO-EDA in GPU4PySCF, on a GPU.

.. grid:: 1 1 2 2

   .. grid-item-card:: Getting Started
      :margin: 0 3 0 0

      A simple introduction

      .. button-link:: ./getting_started/index.html
	 :color: primary
	 :expand:

         To the Getting Started Guide

   .. grid-item-card::  User Guide
      :margin: 0 3 0 0

      A complete guide to using this step

      .. button-link:: ./user_guide/index.html
	 :color: primary
	 :expand:

         To the User Guide

   .. grid-item-card::  Developer Guide
      :margin: 0 3 0 0

      Contributing to the code. Fixing bugs, adding functionality

      .. button-link:: ./developer_guide/index.html
	 :color: primary
	 :expand:

         To the Developer Guide

   .. grid-item-card:: API Reference
      :margin: 0 3 0 0

      The API for the Interaction EDA step

      .. button-link:: ./api/index.html
	 :color: primary
	 :expand:

	 To the API Reference.


.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   getting_started/index
   user_guide/index
   developer_guide/index
   api/index

More Information
================
.. toctree::
   :maxdepth: 1
   :titlesonly:

   authors
   history

   Main SEAMM documentation <https://molssi-seamm.github.io>
