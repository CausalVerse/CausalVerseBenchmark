# Data Modules

This directory contains dataset definitions and the dataset factory used by the CRL codebase.

## Main Files

- `data_factory.py`: central dataset construction logic that maps dataset names to the dataset
  wrappers in this directory.
- `video_dataset.py`: shared video and latent-loading utilities used by the dataset wrappers.
- `fall_dataset.py`: dataset wrapper for the Dynamic Physical Simulation domain scenes
  `fall_simple` and `fall_complex`.
- `projectile_dataset.py`: dataset wrapper for the Dynamic Physical Simulation domain scenes
  `projectile_simple` and `projectile_complex`.
- `collision_dataset.py`: dataset wrapper for the Dynamic Physical Simulation domain scenes
  `collision_simple` and `collision_complex`.
- `fixed_robotics_dataset.py`: dataset wrapper for the Robotic Manipulations domain scenes
  `kitchen`, `living`, `general`, and `study`.

## Usage

Most training and evaluation code does not instantiate dataset classes directly.
Instead, it goes through `DataFactory.create_dataset(...)` and selects a dataset by name.
When loading precomputed latents, the target latent subdirectory is not fixed in the dataset class:
it can be passed through the `latent_view` argument, whose default is `front`.

## Notes

- Dataset roots are configured by the shell scripts under `scripts/`.
- This directory only keeps the dataset files used by the current CRL setup.
- The dataset-specific vector construction also determines which dimensions can later be recovered
  or evaluated. In practice, you can customize which dimensions or factor subsets are used for
  recovery and evaluation, depending on the protocol you want to report.
- The same structure can be extended to other domains as well. For example, a new dataset wrapper
  can be added here for domains such as Traffic Situation Analysis, as long as it exposes the
  required video or latent inputs together with a factor vector suitable for evaluation.
