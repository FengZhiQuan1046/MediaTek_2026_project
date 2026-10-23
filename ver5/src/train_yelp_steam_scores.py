"""Score/log-only entry point; never saves trained weights or sample reports."""
from pathlib import Path
import sys
import json

from src.evaluation_outputs import EvaluationWriter


def main():
    from src import train_mamba_rl as trainer
    from src.train_yelp_steam import main as run

    # Apply only to the current run directory; input data and shared caches are
    # unaffected. These optional reports are omitted rather than deleted later.
    args = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-run-dir', required=True)
    options, _ = parser.parse_known_args(args)
    output = Path(options.output_run_dir).resolve()
    original_path = trainer.Path

    class ScorePath(type(Path())):
        def write_text(self, data, *args, **kwargs):
            if self.parent.resolve() == output and self.name in {
                'recommendations.json', 'preference_analysis.json'
            }:
                return 0
            return super().write_text(data, *args, **kwargs)

    original_parse = trainer.parse_args
    original_metric_block = trainer.log_metric_block
    original_loss_curve = trainer.save_loss_curve
    writer = None

    def parse_args():
        nonlocal writer
        parsed = original_parse()
        output.mkdir(parents=True, exist_ok=True)
        (output / 'config.json').write_text(
            json.dumps(vars(parsed), indent=2, ensure_ascii=False), encoding='utf-8'
        )
        writer = EvaluationWriter(output, parsed.dataset)
        return parsed

    def metric_block(logger, label, metrics, stage, epoch, step):
        # Persist the original metrics before formatting the readable log.
        writer.write(label, metrics, stage, epoch, step)
        original_metric_block(logger, label, metrics, stage, epoch, step)

    trainer.parse_args = parse_args
    trainer.log_metric_block = metric_block
    trainer.Path = ScorePath
    trainer.save_loss_curve = lambda *args, **kwargs: None
    # Last CLI value wins; callers cannot accidentally enable weight saving.
    sys.argv.extend(['--no-save-model-weights', '--no-generate-reasons'])
    try:
        run()
    finally:
        trainer.Path = original_path
        trainer.parse_args = original_parse
        trainer.log_metric_block = original_metric_block
        trainer.save_loss_curve = original_loss_curve


if __name__ == '__main__':
    main()
