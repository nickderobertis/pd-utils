.. pd-utils documentation master file, created by
   cookiecutter-pypi-sphinx.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Pandas Utilities documentation!
********************************************************************

High-level tools for common Pandas workflows

To get started, look here.

.. toctree::
   :caption: Tutorial

   tutorial
   auto_examples/index

An overview of ``pd_utils``
=============================

This is a collection of various functions to extend ``pandas``. Here is a listing
of the current functions.

Quick Links
------------

Find the source code `on Github <https://github.com/nickderobertis/pd-utils>`_.


Merge
----------

.. autosummarynameonly::

      pd_utils.merge.left_merge_latest
      pd_utils.merge.groupby_merge
      pd_utils.merge.groupby_index
      pd_utils.merge.apply_func_to_unique_and_merge


Date-time Handling
--------------------

.. autosummarynameonly::

      pd_utils.datetime_utils.convert_sas_date_to_pandas_date
      pd_utils.datetime_utils.year_month_from_date
      pd_utils.datetime_utils.expand_time
      pd_utils.datetime_utils.expand_months
      pd_utils.datetime_utils.tradedays
      pd_utils.datetime_utils.USTradingCalendar


Fill Data
-----------

.. autosummarynameonly::

      pd_utils.filldata.fillna_by_groups
      pd_utils.filldata.fillna_by_groups_and_keep_one_per_group
      pd_utils.filldata.add_missing_group_rows
      pd_utils.filldata.fill_excluded_rows


Transform
------------

.. autosummarynameonly::

      pd_utils.transform.averages
      pd_utils.transform.state_abbrev
      pd_utils.transform.long_to_wide
      pd_utils.transform.winsorize
      pd_utils.transform.var_change_by_groups
      pd_utils.transform.join_col_strings


Portfolios
------------

.. autosummarynameonly::

      pd_utils.port.portfolio
      pd_utils.port.portfolio_averages
      pd_utils.port.long_short_portfolio


Correlations
--------------

.. autosummarynameonly::

      pd_utils.corr.formatted_corr_df


Cumulate
------------

.. autosummarynameonly::

      pd_utils.cum.cumulate


Regressions
-------------

.. autosummarynameonly::

      pd_utils.regby.reg_by

Querying
-----------

.. autosummarynameonly::

      pd_utils.query.select_rows_by_condition_on_columns
      pd_utils.query.sql

Loading Data
--------------

.. autosummarynameonly::

      pd_utils.load.load_sas

Testing
---------

.. autosummarynameonly::

      pd_utils.testing.to_copy_paste

.. toctree:: api/modules
   :caption: API Documentation
   :maxdepth: 3

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
