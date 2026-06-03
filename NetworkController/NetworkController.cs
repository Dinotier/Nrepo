/*
 * NetworkController.cs — C# Schaltzentrale
 *
 * Entry points:
 *   benchA(BenchParams)      → compact_weights_test --bench-a  (Quake-raw)
 *   benchB(BenchParams)      → compact_weights_test --bench-b  (__frsqrt_rn)
 *   pyTorch_nv(NetParams)    → python train.py --backend cuda
 *   pyTorch_i86(NetParams)   → python train.py --backend cpu
 *   pyTorch_risc(NetParams)  → python train.py --backend mps
 *
 * All three PyTorch backends are always registered — even if unavailable
 * the call logs an informative message and returns cleanly.
 * Communication with subprocesses: JSON params written to stdin, stdout streamed.
 */

using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;

namespace NetworkController
{
    public record BenchParams(
        bool   UseLegacy  = false,
        int    Iterations = 10
    );

    public record NetParams(
        int    Epochs    = 10,
        float  Lr        = 0.01f,
        int    BatchSize = 32,
        string DataDir   = "data"
    );

    public static class Program
    {
        private static readonly string ProjectRoot =
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));

        public static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: NetworkController <command> [options]");
                Console.WriteLine("Commands: benchA  benchB  pyTorch_nv  pyTorch_i86  pyTorch_risc");
                return;
            }

            NetParams net = ParseNetParams(args[1..]);

            switch (args[0])
            {
                case "benchA":       benchA(new BenchParams());  break;
                case "benchB":       benchB(new BenchParams());  break;
                case "pyTorch_nv":   pyTorch_nv(net);            break;
                case "pyTorch_i86":  pyTorch_i86(net);           break;
                case "pyTorch_risc": pyTorch_risc(net);          break;
                default:
                    Console.Error.WriteLine($"[NetworkController] unknown command: {args[0]}");
                    break;
            }
        }

        // ── Benchmark entry points ──────────────────────────────────────────

        public static void benchA(BenchParams p)
        {
            string bin = Path.Combine(ProjectRoot, "build", "compact_weights_test");
            RunProcess(bin, "--bench-a", stdinJson: null);
        }

        public static void benchB(BenchParams p)
        {
            string bin = Path.Combine(ProjectRoot, "build", "compact_weights_test");
            RunProcess(bin, "--bench-b", stdinJson: null);
        }

        // ── PyTorch backends — always three, never crash on unavailability ──

        public static void pyTorch_nv(NetParams p)
            => RunPython(p, "--backend cuda");

        public static void pyTorch_i86(NetParams p)
            => RunPython(p, "--backend cpu");

        public static void pyTorch_risc(NetParams p)
            => RunPython(p, "--backend mps");

        // ── Internal helpers ────────────────────────────────────────────────

        private static void RunPython(NetParams p, string backendArg)
        {
            string script = Path.Combine(ProjectRoot, "emoji_net", "train.py");
            string args   = $"\"{script}\" {backendArg}"
                          + $" --epochs {p.Epochs}"
                          + $" --lr {p.Lr}"
                          + $" --batch-size {p.BatchSize}"
                          + $" --data-dir \"{p.DataDir}\"";
            RunProcess("python3", args, stdinJson: JsonSerializer.Serialize(p));
        }

        private static void RunProcess(string exe, string arguments, string? stdinJson)
        {
            Console.WriteLine($"[NetworkController] {exe} {arguments}");

            var psi = new ProcessStartInfo
            {
                FileName               = exe,
                Arguments              = arguments,
                UseShellExecute        = false,
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                RedirectStandardInput  = stdinJson is not null,
            };

            using var proc = new Process { StartInfo = psi };
            proc.OutputDataReceived += (_, e) => { if (e.Data is not null) Console.WriteLine("[OUT] " + e.Data); };
            proc.ErrorDataReceived  += (_, e) => { if (e.Data is not null) Console.Error.WriteLine("[ERR] " + e.Data); };

            try
            {
                proc.Start();
                proc.BeginOutputReadLine();
                proc.BeginErrorReadLine();

                if (stdinJson is not null)
                {
                    proc.StandardInput.WriteLine(stdinJson);
                    proc.StandardInput.Close();
                }

                proc.WaitForExit();
                Console.WriteLine($"[NetworkController] exit {proc.ExitCode}");
            }
            catch (Exception ex)
            {
                /* Backend unavailable (binary not found, python not installed, etc.)
                 * Log and return — never throw, all three backends must remain callable. */
                Console.Error.WriteLine($"[NetworkController] backend unavailable ({exe}): {ex.Message}");
            }
        }

        private static NetParams ParseNetParams(string[] args)
        {
            int    epochs  = 10;
            float  lr      = 0.01f;
            int    batch   = 32;
            string dataDir = "data";

            for (int i = 0; i + 1 < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--epochs":     int.TryParse  (args[i+1], out epochs);  break;
                    case "--lr":         float.TryParse(args[i+1], out lr);      break;
                    case "--batch-size": int.TryParse  (args[i+1], out batch);   break;
                    case "--data-dir":   dataDir = args[i+1];                    break;
                }
            }
            return new NetParams(epochs, lr, batch, dataDir);
        }
    }
}
