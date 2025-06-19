// File: src/App.js
import React, { useState, useMemo, useRef, useEffect } from 'react';
import { CssBaseline, Container, Box, Button, Grid, LinearProgress, Typography } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import axios from 'axios';
import Cookies from 'js-cookie';
import getTheme from './theme';
import NavBar from './components/NavBar';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import SummaryLengthSelector from './components/SummaryLengthSelector';
import TextInput from './components/TextInput';
import SummaryDisplay from './components/SummaryDisplay';
import Notification from './components/Notification';
import Footer from './components/Footer';

function App() {
  // ✅ Theme with cookies
  const [mode, setMode] = useState(() => Cookies.get('themeMode') || 'light');
  const theme = useMemo(() => getTheme(mode), [mode]);
  const toggleMode = () => {
    setMode(prev => {
      const newMode = prev === 'light' ? 'dark' : 'light';
      Cookies.set('themeMode', newMode, { expires: 30 });
      return newMode;
    });
  };

  // ✅ States
  const [text, setText] = useState('');
  const [length, setLength] = useState('medium');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [notif, setNotif] = useState({ open: false, message: '', severity: 'info' });
  const timerRef = useRef(null);

  const [wordCount, setWordCount] = useState(0);
  const [grammarIssues, setGrammarIssues] = useState([]);
  const [grammarMatches, setGrammarMatches] = useState([]);

  // ✅ Grammar check helper
  const updateTextStats = async (inputText) => {
    const wc = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
    setWordCount(wc);

    try {
      const { data } = await axios.post(
        "https://api.languagetool.org/v2/check",
        new URLSearchParams({
          text: inputText,
          language: "en-US"
        })
      );
      console.log("Grammar Matches:", data.matches);
      setGrammarIssues(data.matches || []);
      setGrammarMatches(data.matches || []);
    } catch (err) {
      console.error("Grammar check failed", err);
      setGrammarIssues([]);
      setGrammarMatches([]);
    }
  };

  // ✅ Auto-correct
  const applyAutoCorrection = () => {
    let newText = text;
    let offsetShift = 0;

    const sorted = [...grammarMatches].sort((a, b) => a.offset - b.offset);

    sorted.forEach(match => {
      if (match.replacements && match.replacements.length > 0) {
        const replacement = match.replacements[0].value;
        const start = match.offset + offsetShift;
        const end = start + match.length;

        newText = newText.slice(0, start) + replacement + newText.slice(end);
        offsetShift += replacement.length - match.length;
      }
    });

    setText(newText);
    updateTextStats(newText);
    setNotif({ open: true, message: 'Auto-correction applied!', severity: 'success' });
  };

  // ✅ Backend cancel API call
  const cancelSummarization = async () => {
    try {
      await axios.post('http://localhost:8000/api/summarize/cancel');
      console.log("Cancel request sent to backend");
    } catch (err) {
      console.error("Cancel request failed", err);
    }
  };

  // ✅ Manual cancel button handler with confirmation
  const handleManualCancel = async () => {
    const confirm = window.confirm("Are you sure you want to cancel the summarization process?");
    if (confirm) {
      clearInterval(timerRef.current);
      await cancelSummarization();
      setLoading(false);
      setProgress(0);
      setStatusMsg('Summarization cancelled.');
      setNotif({ open: true, message: 'Summarization cancelled.', severity: 'warning' });
    }
  };

  // ✅ Refresh warning + backend cancel
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (loading) {
        e.preventDefault();
        e.returnValue = 'Are you sure you want to leave? This will stop the summarization.';
        cancelSummarization();
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [loading]);

  // ✅ Summarize handler
  const handleSummarize = async (inputText) => {
    setLoading(true);
    setProgress(0);
    setStatusMsg('Initializing summary…');
    timerRef.current = setInterval(() => {
      setProgress(prev => {
        const next = Math.min(prev + Math.random() * 10, 80);
        if (next > 50) setStatusMsg('Generating summary…');
        return next;
      });
    }, 500);

    try {
      const { data } = await axios.post(
        'http://localhost:8000/api/summarize/',
        { text: inputText, summary_length: length }
      );
      clearInterval(timerRef.current);
      setProgress(100);
      setStatusMsg('Finalizing…');
      await new Promise(res => setTimeout(res, 300));
      setSummary(data.summary);
      setNotif({ open: true, message: 'Summary generated successfully!', severity: 'success' });
    } catch (err) {
      clearInterval(timerRef.current);
      console.error(err);
      setNotif({ open: true, message: 'Failed to generate summary.', severity: 'error' });
    } finally {
      setLoading(false);
      setTimeout(() => setProgress(0), 300);
    }
  };

  // ✅ File summarizer (unchanged)
  const handleFileSummarize = async (file) => {
    setLoading(true);
    setProgress(0);
    setStatusMsg('Uploading file…');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('summary_length', length);

    try {
      const { data } = await axios.post(
        'http://localhost:8000/api/summarize/file',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: evt => {
            const pct = Math.round((evt.loaded * 100) / evt.total);
            setProgress(pct * 0.5);
            setStatusMsg(`Uploading… ${pct}%`);
          }
        }
      );
      setProgress(80);
      setStatusMsg('Summarizing file…');
      await new Promise(res => setTimeout(res, 500));
      setSummary(data.summary);
      setNotif({ open: true, message: 'Summary generated!', severity: 'success' });
    } catch (err) {
      console.error(err);
      setNotif({ open: true, message: 'File summarization failed.', severity: 'error' });
    } finally {
      setLoading(false);
      setTimeout(() => setProgress(0), 300);
    }
  };

  // ✅ Text & file handler
  const handleFrontText = (parsedText, file) => {
    if (parsedText) {
      setText(parsedText);
      updateTextStats(parsedText);
    } else if (file) {
      handleFileSummarize(file);
    }
  };

  const handleTextChange = (val) => {
    setText(val);
    updateTextStats(val);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <NavBar mode={mode} toggleMode={toggleMode} />
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Header />

        {loading && (
          <Box sx={{ width: '100%', mb: 2 }}>
            <Typography variant="body2" align="center">{statusMsg}</Typography>
            <LinearProgress variant="determinate" value={progress} />
          </Box>
        )}

        <Box mt={4}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FileUpload onText={handleFrontText} />
            </Grid>
            <Grid item xs={12} md={6}>
              <SummaryLengthSelector
                length={length}
                setLength={setLength}
                options={[
                  { value: 'short', label: 'Short' },
                  { value: 'medium', label: 'Moderate' },
                  { value: 'detailed', label: 'Detailed' }
                ]}
              />
            </Grid>
            <Grid item xs={12}>
              <TextInput text={text} setText={handleTextChange} rows={12} />
            </Grid>

            {wordCount > 0 && (
              <Grid item xs={12}>
                <Typography variant="body2">Word count: {wordCount}</Typography>
                <Typography variant="body2">Grammar issues: {grammarIssues.length}</Typography>
                {grammarIssues.slice(0, 5).map((issue, idx) => (
                  <Typography key={idx} variant="caption" color="error">
                    ⚠️ {issue.message} (at: {issue.context?.text})
                  </Typography>
                ))}
                {grammarMatches.length > 0 && grammarMatches.some(m => m.replacements?.length) && (
                  <Button
                    variant="outlined"
                    color="secondary"
                    sx={{ mt: 1, mr: 1 }}
                    onClick={applyAutoCorrection}
                  >
                    Auto-correct Grammar Issues
                  </Button>
                )}
                {loading && (
                  <Button
                    variant="outlined"
                    color="error"
                    sx={{ mt: 1 }}
                    onClick={handleManualCancel}
                  >
                    Cancel Summarization
                  </Button>
                )}
              </Grid>
            )}

            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                onClick={() => handleSummarize(text)}
                disabled={loading}
                fullWidth
              >
                {loading ? 'Summarizing…' : 'Summarize'}
              </Button>
            </Grid>
            <Grid item xs={12}>
              <SummaryDisplay summary={summary} />
            </Grid>
          </Grid>
        </Box>
      </Container>

      <Footer />
      <Notification
        open={notif.open}
        message={notif.message}
        severity={notif.severity}
        onClose={() => setNotif({ ...notif, open: false })}
      />
    </ThemeProvider>
  );
}

export default App;
