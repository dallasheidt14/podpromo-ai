import { beforeAll, afterAll, afterEach, vi, it, expect } from "vitest";
import { setupServer } from "msw/node";
import { rest } from "msw";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import EpisodeUpload from "../../components/EpisodeUpload";
import ClipGallery from "../../components/ClipGallery";

const EP = "ep-yt-1";
let calls = 0;

const server = setupServer(
  rest.post("/api/upload-youtube", async (_req, res, ctx) =>
    res(ctx.json({ episode_id: EP }))
  ),
  rest.get(`/api/progress/${EP}`, async (_req, res, ctx) => {
    calls++;
    if (calls < 2) return res(ctx.json({ stage: "scoring", percent: 70, message: "Scoring…" }));
    return res(ctx.json({ stage: "completed", percent: 100, message: "Done" }));
  }),
  rest.get(`/api/episodes/${EP}/clips`, async (_req, res, ctx) =>
    res(ctx.json([{ id: "c1", start: 0, end: 8, title: "Demo", preview_url: "/clips/previews/demo.mp4" }]))
  ),
);

beforeAll(() => server.listen());
afterEach(() => { server.resetHandlers(); calls = 0; });
afterAll(() => server.close());

it("auto-refreshes clips after YouTube processing completes", async () => {
  const onStart = vi.fn();
  const onClipsFetched = vi.fn();
  
  render(
    <div>
      <EpisodeUpload onStart={onStart} />
      <ClipGallery episodeId={EP} onClipsFetched={onClipsFetched} />
    </div>
  );

  // switch to YT and submit
  fireEvent.click(screen.getByText(/YouTube Link/i));
  const input = screen.getByPlaceholderText(/https:\/\/www\.youtube\.com|youtu\.be/i);
  fireEvent.change(input, { target: { value: "https://youtu.be/abc" } });
  fireEvent.click(screen.getByText(/Start from YouTube/i));

  await waitFor(() => expect(onStart).toHaveBeenCalledWith(EP));

  // clips arrive after progress completes — no F5 required
  await waitFor(() => expect(screen.getByText("Demo")).toBeInTheDocument(), { timeout: 5000 });
  
  // Verify clips were fetched
  expect(onClipsFetched).toHaveBeenCalledWith(
    expect.arrayContaining([
      expect.objectContaining({
        id: "c1",
        title: "Demo",
        previewUrl: "/clips/previews/demo.mp4"
      })
    ])
  );
});
