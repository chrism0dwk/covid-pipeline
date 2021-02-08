# Cronjob wrapper for Lancaster Stochastic SEIR COVID model

The `enqueue_pipeline.sh` script is designed to run as a cronjob, and
takes one argument, ie. the date (in ISO8601) of the analysis. e.g.

```bash
$ . enqueue_pipeline.sh 2021-02-08
```

The script submits a job to a SGE-based cluster, with a
`qsub -sync y` synchronous call.  The script waits for the job
to be enqueued and subsequently run, before copying results from
a scratch directory to a long-term storage destination.
