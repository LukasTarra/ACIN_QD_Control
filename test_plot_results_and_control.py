import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from main import plot_results_and_control

class TestPlotResultsAndControl(unittest.TestCase):

    def setUp(self):
        # Create a mock results object
        self.mock_results = MagicMock()
        self.mock_results.times = np.linspace(0, 10, 100)
        self.mock_results.expect = [np.random.rand(100) for _ in range(6)]

        # Define a simple control function
        self.control_fun = lambda t: np.sin(t)
        self.t_array = np.linspace(0, 10, 100)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.show')
    def test_plot_results_and_control_normal_operation(self, mock_show, mock_grid, mock_legend, mock_title, mock_ylabel, mock_xlabel, mock_plot, mock_figure):
        # Test normal operation of the function
        plot_results_and_control(self.mock_results, self.control_fun, self.t_array)

        # Assert that the figure and plot functions were called
        mock_figure.assert_called_once_with(figsize=(10, 6))
        self.assertEqual(mock_plot.call_count, 6)
        mock_xlabel.assert_called_once_with("Time (ps)")
        mock_ylabel.assert_called_once_with("Population")
        mock_title.assert_called_once_with("Population Trajectories")
        mock_legend.assert_called_once()
        mock_grid.assert_called_once()
        mock_show.assert_called_once()

        # Assert that the control field plot functions were called
        mock_figure.assert_called_with()
        mock_plot.assert_called_with(self.t_array, self.control_fun(self.t_array))
        mock_xlabel.assert_called_with("Time (ps)")
        mock_ylabel.assert_called_with("Control Field (meV)")
        mock_title.assert_called_with("Control Field")
        mock_show.assert_called()

        # Assert that the FFT plot functions were called
        control_FF_array = self.control_fun(self.t_array)
        control_FF_FFT = np.abs(np.fft.fft(control_FF_array))[:len(control_FF_array)//2]
        mock_figure.assert_called_with()
        mock_plot.assert_called_with(control_FF_FFT)
        mock_xlabel.assert_called_with("Frequency")
        mock_ylabel.assert_called_with("Amplitude")
        mock_title.assert_called_with("Control Field FFT")
        mock_show.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.show')
    def test_plot_results_and_control_empty_results(self, mock_show, mock_grid, mock_legend, mock_title, mock_ylabel, mock_xlabel, mock_plot, mock_figure):
        # Test the function with empty results
        empty_results = MagicMock()
        empty_results.times = np.array([])
        empty_results.expect = [np.array([]) for _ in range(6)]

        plot_results_and_control(empty_results, self.control_fun, self.t_array)

        # Assert that the figure and plot functions were called with empty data
        mock_figure.assert_called_once_with(figsize=(10, 6))
        self.assertEqual(mock_plot.call_count, 6)
        mock_xlabel.assert_called_once_with("Time (ps)")
        mock_ylabel.assert_called_once_with("Population")
        mock_title.assert_called_once_with("Population Trajectories")
        mock_legend.assert_called_once()
        mock_grid.assert_called_once()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.show')
    def test_plot_results_and_control_zero_control_field(self, mock_show, mock_grid, mock_legend, mock_title, mock_ylabel, mock_xlabel, mock_plot, mock_figure):
        # Test the function with a zero control field
        zero_control_fun = lambda t: np.zeros_like(t)
        plot_results_and_control(self.mock_results, zero_control_fun, self.t_array)

        # Assert that the figure and plot functions were called
        mock_figure.assert_called_once_with(figsize=(10, 6))
        self.assertEqual(mock_plot.call_count, 6)
        mock_xlabel.assert_called_once_with("Time (ps)")
        mock_ylabel.assert_called_once_with("Population")
        mock_title.assert_called_once_with("Population Trajectories")
        mock_legend.assert_called_once()
        mock_grid.assert_called_once()
        mock_show.assert_called_once()

        # Assert that the control field plot functions were called with zero data
        mock_figure.assert_called_with()
        mock_plot.assert_called_with(self.t_array, zero_control_fun(self.t_array))
        mock_xlabel.assert_called_with("Time (ps)")
        mock_ylabel.assert_called_with("Control Field (meV)")
        mock_title.assert_called_with("Control Field")
        mock_show.assert_called()

        # Assert that the FFT plot functions were called with zero data
        control_FF_array = zero_control_fun(self.t_array)
        control_FF_FFT = np.abs(np.fft.fft(control_FF_array))[:len(control_FF_array)//2]
        mock_figure.assert_called_with()
        mock_plot.assert_called_with(control_FF_FFT)
        mock_xlabel.assert_called_with("Frequency")
        mock_ylabel.assert_called_with("Amplitude")
        mock_title.assert_called_with("Control Field FFT")
        mock_show.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.show')
    def test_plot_results_and_control_large_control_field(self, mock_show, mock_grid, mock_legend, mock_title, mock_ylabel, mock_xlabel, mock_plot, mock_figure):
        # Test the function with a large control field
        large_control_fun = lambda t: 1e6 * np.sin(t)
        plot_results_and_control(self.mock_results, large_control_fun, self.t_array)

        # Assert that the figure and plot functions were called
        mock_figure.assert_called_once_with(figsize=(10, 6))
        self.assertEqual(mock_plot.call_count, 6)
        mock_xlabel.assert_called_once_with("Time (ps)")
        mock_ylabel.assert_called_once_with("Population")
        mock_title.assert_called_once_with("Population Trajectories")
        mock_legend.assert_called_once()
        mock_grid.assert_called_once()
        mock_show.assert_called_once()

        # Assert that the control field plot functions were called with large data
        mock_figure.assert_called_with()
        mock_plot.assert_called_with(self.t_array, large_control_fun(self.t_array))
        mock_xlabel.assert_called_with("Time (ps)")
        mock_ylabel.assert_called_with("Control Field (meV)")
        mock_title.assert_called_with("Control Field")
        mock_show.assert_called()

        # Assert that the FFT plot functions were called with large data
        control_FF_array = large_control_fun(self.t_array)
        control_FF_FFT = np.abs(np.fft.fft(control_FF_array))[:len(control_FF_array)//2]
        mock_figure.assert_called_with()
        mock_plot.assert_called_with(control_FF_FFT)
        mock_xlabel.assert_called_with("Frequency")
        mock_ylabel.assert_called_with("Amplitude")
        mock_title.assert_called_with("Control Field FFT")
        mock_show.assert_called()

if __name__ == '__main__':
    unittest.main()