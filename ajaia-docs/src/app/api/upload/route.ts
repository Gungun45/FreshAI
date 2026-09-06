import { NextRequest, NextResponse } from 'next/server';
import { parseUploadedBuffer } from '@/lib/fileParsers';
import { Storage } from '@/lib/storage';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File | null;
    const ownerId = formData.get('ownerId') as string | null;
    const createNewDoc = formData.get('createNewDoc') === 'true';

    if (!file) {
      return NextResponse.json({ success: false, error: 'No file uploaded' }, { status: 400 });
    }

    const filename = file.name;
    const mimeType = file.type || 'text/plain';
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const parsed = await parseUploadedBuffer(buffer, filename, mimeType);

    if (createNewDoc && ownerId) {
      const doc = Storage.create({
        title: parsed.title,
        content: parsed.content,
        ownerId: ownerId,
      });

      return NextResponse.json({
        success: true,
        filename,
        title: parsed.title,
        content: parsed.content,
        format: parsed.format,
        createdDocument: doc,
      });
    }

    return NextResponse.json({
      success: true,
      filename,
      title: parsed.title,
      content: parsed.content,
      format: parsed.format,
    });
  } catch (error: any) {
    console.error('File upload error:', error);
    return NextResponse.json({ success: false, error: error.message || 'File processing failed' }, { status: 500 });
  }
}
